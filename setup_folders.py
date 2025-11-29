import os

# Các folder con muốn tạo
folders = [
    "src",
    "notebooks",
    "data_raw",
    "data_npy",
    "meta",
]

# Tạo file rỗng trong src để thành package Python
files = {
    "src/__init__.py": "",
    "src/preprocess.py": "",
    "src/dataset.py": "",
    "src/model.py": "",
    "src/train_utils.py": "",
    "notebooks/01_explore_dataset.ipynb": "",
    "notebooks/02_preprocess_demo.ipynb": "",
    "notebooks/03_train_word_level.ipynb": "",
    "meta/labels.txt": "",
    "meta/filelist.csv": "",
    "requirements.txt": "",
    ".gitignore": "",
}

def main():
    # Tạo folder
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # Tạo file rỗng
    for path, content in files.items():
        # Đảm bảo folder cha tồn tại
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Nếu file chưa tồn tại thì tạo
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created file: {path}")
        else:
            print(f"File already exists: {path}")

if __name__ == "__main__":
    main()
