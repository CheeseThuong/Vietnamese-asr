"""
rename_files.py — Chuẩn hóa tên file/folder tiếng Việt
=========================================================
Quét dự án, chuyển tên file có dấu → không dấu, thay space → '_', viết thường.
Tự động cập nhật tham chiếu trong source code.

Sử dụng:
    python rename_files.py              # Dry-run (chỉ hiển thị, không đổi tên)
    python rename_files.py --execute    # Thực sự đổi tên + cập nhật tham chiếu
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

# ======================================================================
# CẤU HÌNH
# ======================================================================

# Thư mục sẽ bỏ qua (không quét)
SKIP_DIRS = {
    ".git", ".conda", "__pycache__", "node_modules", ".vscode",
    "venv", ".venv", "env", ".env", ".idea", "dist", "build",
    "processed_data_merged", "processed_data_vivos", "processed_data_vlsp",
    "final_model", "kaggle_output", "token", "certs",
    "Data", "data", "uploads", "results", "models",
}

# Extension của source code sẽ được cập nhật tham chiếu
SOURCE_EXTENSIONS = {".py", ".html", ".js", ".json", ".yaml", ".yml", ".cfg", ".md", ".txt", ".css", ".bat", ".sh"}

# Extension/filename sẽ không đổi tên (binary/data files lớn)
SKIP_RENAME_EXTENSIONS = {".safetensors", ".bin", ".pkl", ".h5", ".onnx", ".pt", ".pth"}

# Files sẽ không đổi tên (Python special files, standard project files)
SKIP_FILENAMES = {
    "__init__.py", "__main__.py", "__all__.py", "_version.py",
    "README.md", "LICENSE", "CHANGELOG.md", "Makefile",
    "Dockerfile", "CMakeLists.txt", ".gitignore", ".gitkeep",
    "setup.py", "setup.cfg", "pyproject.toml",
}

# Patterns sẽ không đổi tên
SKIP_FILENAME_PATTERNS = [
    r"^__.*__\.py$",           # Python dunder files
    r"^\..*$",                 # Dotfiles
    r"^CMakeLists\.txt$",      # CMake (case-sensitive)
]


# ======================================================================
# CHUYỂN ĐỔI TIẾNG VIỆT KHÔNG DẤU
# ======================================================================

# Bảng chuyển đổi đặc biệt cho tiếng Việt (unicodedata không handle hết)
_VN_SPECIAL = {
    "đ": "d", "Đ": "D",
    "ơ": "o", "Ơ": "O",
    "ư": "u", "Ư": "U",
}

def remove_vietnamese_accents(text: str) -> str:
    """Chuyển tiếng Việt có dấu sang không dấu."""
    # Thay thế ký tự đặc biệt trước
    for vn_char, ascii_char in _VN_SPECIAL.items():
        text = text.replace(vn_char, ascii_char)

    # Dùng unicodedata decompose rồi loại bỏ combining marks
    nfkd = unicodedata.normalize("NFD", text)
    result = ""
    for ch in nfkd:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):  # Mark (combining accent)
            continue
        result += ch

    return result


def sanitize_filename(name: str) -> str:
    """
    Chuẩn hóa tên file:
    - Bỏ dấu tiếng Việt
    - Thay space/tab → '_'
    - Loại bỏ ký tự đặc biệt (giữ lại a-z, 0-9, _, -, .)
    - Viết thường
    """
    name = remove_vietnamese_accents(name)
    name = name.lower()
    name = re.sub(r"[\s\t]+", "_", name)
    name = re.sub(r"[^a-z0-9_\-.]", "", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def needs_rename(name: str) -> bool:
    """Kiểm tra tên file có cần đổi không."""
    # Không đổi tên các file đặc biệt
    if name in SKIP_FILENAMES:
        return False
    for pattern in SKIP_FILENAME_PATTERNS:
        if re.match(pattern, name):
            return False
    return sanitize_filename(name) != name


# ======================================================================
# QUÉT VÀ THU THẬP FILE CẦN ĐỔI
# ======================================================================

def collect_rename_candidates(project_root: Path):
    """
    Quét project_root và trả về danh sách (old_path, new_path).
    Quét từ sâu → nông (bottom-up) để đổi tên file trước, folder sau.
    """
    candidates = []
    all_items = []

    for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
        # Loại bỏ thư mục skip
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        dp = Path(dirpath)

        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SKIP_RENAME_EXTENSIONS:
                continue
            if needs_rename(fname):
                old = dp / fname
                new_name = sanitize_filename(fname)
                new = dp / new_name
                all_items.append((old, new))

        for dname in dirnames:
            if needs_rename(dname):
                old = dp / dname
                new_name = sanitize_filename(dname)
                new = dp / new_name
                all_items.append((old, new))

    # Sắp xếp: file trước, folder sau; sâu trước, nông sau
    all_items.sort(key=lambda x: (-len(x[0].parts), x[0].is_dir()))

    return all_items


# ======================================================================
# CẬP NHẬT THAM CHIẾU TRONG SOURCE CODE
# ======================================================================

def find_source_files(project_root: Path):
    """Tìm tất cả file source code cần cập nhật tham chiếu."""
    source_files = []
    for dirpath, dirnames, filenames in os.walk(project_root, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in SOURCE_EXTENSIONS:
                source_files.append(Path(dirpath) / fname)
    return source_files


def update_references(source_files: list, old_name: str, new_name: str, dry_run: bool, log_lines: list):
    """Tìm và thay thế tham chiếu old_name → new_name trong source files."""
    updated_files = []

    if old_name == new_name:
        return updated_files

    for src_file in source_files:
        try:
            content = src_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if old_name not in content:
            continue

        new_content = content.replace(old_name, new_name)

        if new_content != content:
            if not dry_run:
                src_file.write_text(new_content, encoding="utf-8")
            updated_files.append(str(src_file))

    return updated_files


# ======================================================================
# MAIN
# ======================================================================

def main():
    dry_run = "--execute" not in sys.argv

    # Xác định project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir  # Script nằm ở gốc project

    print("=" * 70)
    mode_str = "DRY-RUN (chi hien thi, khong doi ten)" if dry_run else "EXECUTE (doi ten that)"
    print(f"  Rename Files - Chuan hoa ten file tieng Viet")
    print(f"  Mode: {mode_str}")
    print(f"  Project: {project_root}")
    print("=" * 70)

    # Thu thập file cần đổi
    candidates = collect_rename_candidates(project_root)

    if not candidates:
        print("\n  Khong tim thay file/folder nao can doi ten!")
        print("  Tat ca ten file da chuan hoa.")
        return

    print(f"\n  Tim thay {len(candidates)} file/folder can doi ten:\n")

    log_lines = []
    source_files = find_source_files(project_root) if candidates else []

    for i, (old_path, new_path) in enumerate(candidates, 1):
        old_name = old_path.name
        new_name = new_path.name
        rel_old = old_path.relative_to(project_root)
        rel_new = new_path.relative_to(project_root)

        type_str = "[DIR ]" if old_path.is_dir() else "[FILE]"
        print(f"  {i:3d}. {type_str} {rel_old}")
        print(f"         -> {rel_new}")

        # Tìm references
        ref_files = update_references(source_files, old_name, new_name, dry_run=True, log_lines=log_lines)
        if ref_files:
            print(f"         (Tham chieu trong {len(ref_files)} file)")

        log_entry = f"{old_path} -> {new_path}"
        if ref_files:
            log_entry += f"\n  References updated in: {', '.join(ref_files)}"
        log_lines.append(log_entry)

    print(f"\n  Tong cong: {len(candidates)} doi tuong can doi ten")

    if dry_run:
        print("\n" + "=" * 70)
        print("  Day la DRY-RUN. De thuc su doi ten, chay:")
        print("    python rename_files.py --execute")
        print("=" * 70)
    else:
        # Xác nhận trước khi thực thi
        print("\n" + "-" * 70)
        answer = input("  Ban co chac muon doi ten tat ca? (y/N): ").strip().lower()
        if answer != "y":
            print("  Da huy. Khong doi ten gi het.")
            return

        print("\n  Dang doi ten...\n")
        executed = 0

        for old_path, new_path in candidates:
            old_name = old_path.name
            new_name = new_path.name

            try:
                # Cập nhật tham chiếu trước
                ref_files = update_references(source_files, old_name, new_name, dry_run=False, log_lines=log_lines)

                # Đổi tên
                if old_path.exists():
                    # Đảm bảo thư mục đích tồn tại
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path.rename(new_path)
                    executed += 1
                    status = "OK"
                else:
                    status = "SKIP (khong ton tai)"

                print(f"  [{status}] {old_path.name} -> {new_path.name}")

            except Exception as e:
                print(f"  [ERROR] {old_path.name}: {e}")

        # Ghi log
        log_path = project_root / "rename_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("Rename Log - Chuan hoa ten file tieng Viet\n")
            f.write("=" * 70 + "\n\n")
            for entry in log_lines:
                f.write(entry + "\n\n")
            f.write(f"\nTong cong: {executed}/{len(candidates)} doi tuong da doi ten\n")

        print(f"\n  Hoan tat! Da doi ten {executed}/{len(candidates)} doi tuong.")
        print(f"  Log file: {log_path}")


if __name__ == "__main__":
    main()
