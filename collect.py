import carla
import time
import cv2
import numpy as np
import pandas as pd
import os
import argparse

# Connect to CARLA
def make_client(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    return client


def save_image(image, save_path):
    """Save CARLA image to disk."""
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]  # drop alpha
    cv2.imwrite(save_path, img)


def main(output_csv, img_folder):
    os.makedirs(img_folder, exist_ok=True)
    client = make_client()
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # RGB camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '360')
    camera_bp.set_attribute('fov', '110')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    data_rows = []
    frame_id = 0

    print("Collecting data — drive manually. Press CTRL+C to stop.")

    def on_image(image):
        nonlocal frame_id

        img_path = os.path.join(img_folder, f"{frame_id:06d}.jpg")
        save_image(image, img_path)

        control = vehicle.get_control()
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5 * 3.6  # km/h

        row = {
            "center_image_path": f"{frame_id:06d}.jpg",
            "steering": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
            "speed": speed
        }
        data_rows.append(row)

        frame_id += 1

    camera.listen(on_image)

    try:
        while True:
            time.sleep(0.05)  # 20Hz
    except KeyboardInterrupt:
        print("\nStopping and saving...")
    finally:
        camera.stop()
        df = pd.DataFrame(data_rows)
        df.to_csv(output_csv, index=False)
        vehicle.destroy()
        camera.destroy()
        print("Saved:", output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/drive_log.csv")
    parser.add_argument("--img_dir", default="data/IMG")
    a = parser.parse_args()

    main(a.csv, a.img_dir)
