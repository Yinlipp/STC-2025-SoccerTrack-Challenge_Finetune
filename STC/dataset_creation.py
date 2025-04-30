import os
import cv2
import pandas as pd
import random
import shutil
import yaml

def create_data_yaml(output_root, class_names=['player']):
    data = {
        'train': os.path.abspath(os.path.join(output_root, 'images/train')),
        'val': os.path.abspath(os.path.join(output_root, 'images/test')),
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ data.yaml created at {yaml_path}")


def convert_mot_to_yolo_with_split(
    mot_txt_path: str,
    video_path: str,
    output_root: str,
    use_id_as_class: bool = False,
    train_ratio: float = 0.7,
    seed: int = 42
):
    """
    Convert MOT-format annotations to YOLO format, extract image frames from a video,
    and split the dataset into training and testing subsets.

    Parameters:
        mot_txt_path (str): Path to the MOT-format annotation file (e.g., 'gt.txt').
        video_path (str): Path to the input video file.
        output_root (str): Root directory for output.
        use_id_as_class (bool): Whether to use object ID as YOLO class label.
        train_ratio (float): Proportion of frames to be used for training (default is 0.7).
        seed (int): Random seed for reproducible train/test split.
    """

    # Temporary folders to hold extracted images and labels
    temp_img_dir = os.path.join(output_root, "images_all")
    temp_lbl_dir = os.path.join(output_root, "labels_all")
    os.makedirs(temp_img_dir, exist_ok=True)
    os.makedirs(temp_lbl_dir, exist_ok=True)

    # Final output folder structure
    for subset in ['train', 'test']:
        os.makedirs(os.path.join(output_root, f"images/{subset}"), exist_ok=True)
        os.makedirs(os.path.join(output_root, f"labels/{subset}"), exist_ok=True)

    # Load MOT annotation
    df = pd.read_csv(mot_txt_path, header=None, usecols=range(6),
                     names=['frame', 'id', 'x', 'y', 'w', 'h'])
    frame_set = sorted(df['frame'].unique())

    # Open video and get image dimensions
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_id = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id not in frame_set:
            continue

        # Save the image frame
        img_name = f"{video_name}_{frame_id:04d}.png"
        img_path = os.path.join(temp_img_dir, img_name)
        cv2.imwrite(img_path, frame)

        # Generate YOLO-format labels
        frame_df = df[df['frame'] == frame_id]
        yolo_lines = []
        for _, row in frame_df.iterrows():
            x_center = (row['x'] + row['w'] / 2) / img_width
            y_center = (row['y'] + row['h'] / 2) / img_height
            width = row['w'] / img_width
            height = row['h'] / img_height
            class_id = int(row['id']) if use_id_as_class else 0
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        label_path = os.path.join(temp_lbl_dir, img_name.replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        saved_frames.append(img_name)

    cap.release()

    # Randomly split frames into train/test sets
    random.seed(seed)
    random.shuffle(saved_frames)
    split_index = int(train_ratio * len(saved_frames))
    train_files = saved_frames[:split_index]
    test_files = saved_frames[split_index:]

    # Move files into train/test folders
    for fname in train_files:
        shutil.move(os.path.join(temp_img_dir, fname), os.path.join(output_root, "images/train", fname))
        shutil.move(os.path.join(temp_lbl_dir, fname.replace('.png', '.txt')), os.path.join(output_root, "labels/train", fname.replace('.png', '.txt')))

    for fname in test_files:
        shutil.move(os.path.join(temp_img_dir, fname), os.path.join(output_root, "images/test", fname))
        shutil.move(os.path.join(temp_lbl_dir, fname.replace('.png', '.txt')), os.path.join(output_root, "labels/test", fname.replace('.png', '.txt')))

    # Remove temporary folders
    os.rmdir(temp_img_dir)
    os.rmdir(temp_lbl_dir)

    print(f"Conversion complete! Total frames: {len(saved_frames)}. "
          f"Training: {len(train_files)}, Testing: {len(test_files)}.")

if __name__ == '__main__':
    # A folder containing both video files and their corresponding annotations in MOT format.
    folder = '/home/y_li/workspace3/STC/Train/'
    # The output root directory where the converted YOLO format dataset will be saved.
    output_root = "/home/y_li/workspace3/STC/STC_YOLO_finetune_dataset/"

    for filename in os.listdir(folder):
        if filename.endswith('.mp4'):

            video_path = os.path.join(folder, filename)
            mot_txt_path = os.path.splitext(video_path)[0] + ".txt"

            convert_mot_to_yolo_with_split(
                mot_txt_path= mot_txt_path,
                video_path= video_path,
                output_root= output_root,
                use_id_as_class=False,
                train_ratio=0.7
            )

    print(f"Dataset successfully generated at {output_root}.")

    # Create data.yaml file
    create_data_yaml(output_root= output_root, class_names=['player'])
    print(f"data.yaml successfully generated at {output_root}.")







