import carla
import torch
import cv2
import numpy as np
import time

from model import NvidiaModel
from utils import preprocess_image, to_tensor, smooth_steering

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_client(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    return client


def main(model_path):
    # Load model
    model = NvidiaModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    client = make_client()
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # RGB camera
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '360')
    cam_bp.set_attribute('fov', '110')
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    last_steer = 0.0

    print("Autonomous mode ON (CTRL+C to stop).")

    def on_image(image):
        nonlocal last_steer

        # Convert CARLA image → numpy
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))[:, :, :3]

        # Preprocess exactly like training
        img_p = preprocess_image(img)
        tensor = to_tensor(img_p).to(DEVICE)

        with torch.no_grad():
            pred = model(tensor).cpu().numpy().flatten()[0]

        # Smooth predictions
        smoothed_steer = smooth_steering(last_steer, pred, factor=0.3)
        last_steer = smoothed_steer

        # Apply control
        control = carla.VehicleControl()
        control.steer = float(smoothed_steer)
        control.throttle = 0.40           # constant throttle for basic test
        control.brake = 0.0

        vehicle.apply_control(control)

    camera.listen(on_image)

    try:
        while True:
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopping autonomy...")
    finally:
        camera.stop()
        vehicle.destroy()
        camera.destroy()
        print("Destroyed actors.")


if __name__ == "__main__":
    main("checkpoints/model_epoch20.pt")
