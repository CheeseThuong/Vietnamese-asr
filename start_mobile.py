"""
🚀 VietASR Pro - Mobile Access Launcher
Tự động tạo SSL certificate và khởi chạy server để truy cập từ điện thoại.

Cách dùng:
    python start_mobile.py

Sau khi chạy, mở trình duyệt trên điện thoại và truy cập URL hiển thị.
"""

import os
import ssl
import sys
import socket
import subprocess
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────
CERT_DIR = Path(__file__).resolve().parent / "certs"
CERT_FILE = CERT_DIR / "server.crt"
KEY_FILE = CERT_DIR / "server.key"
PORT = 5000


def get_local_ip():
    """Tìm địa chỉ IP LAN của máy (không phải 127.0.0.1)."""
    candidates = []

    # Method 1: socket connect (most reliable on Windows)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Method 2: hostname resolution
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                candidates.append(ip)
    except Exception:
        pass

    # Method 3: iterate all interfaces
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if socket.AF_INET in addrs:
                for addr in addrs[socket.AF_INET]:
                    ip = addr.get("addr", "")
                    if ip and not ip.startswith("127."):
                        candidates.append(ip)
    except ImportError:
        pass

    return candidates[0] if candidates else "127.0.0.1"


def generate_ssl_cert():
    """Tạo self-signed SSL certificate nếu chưa có."""
    if CERT_FILE.exists() and KEY_FILE.exists():
        print("✓ SSL certificate đã tồn tại")
        return True

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    print("🔐 Đang tạo SSL certificate (self-signed)...")

    try:
        # Try using Python's built-in ssl/cryptography
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta, timezone
        import ipaddress

        local_ip = get_local_ip()

        # Generate key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Build certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "VietASR Pro Local Server"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VietASR Pro"),
        ])

        san_entries = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
        except Exception:
            pass

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(key, hashes.SHA256())
        )

        # Write files
        with open(KEY_FILE, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        with open(CERT_FILE, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print("✓ SSL certificate đã được tạo thành công!")
        return True

    except ImportError:
        print("⚠ Thư viện 'cryptography' chưa được cài.")
        print("  Đang cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography"])
        print("  ✓ Đã cài xong. Vui lòng chạy lại script này.")
        return False


def print_qr_code(url):
    """In QR code ra terminal (nếu có thư viện qrcode)."""
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=1, border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except ImportError:
        print("  💡 Tip: Cài 'qrcode' để hiển thị QR code:")
        print(f"     pip install qrcode")


def main():
    local_ip = get_local_ip()
    url_https = f"https://{local_ip}:{PORT}"
    url_local = f"https://localhost:{PORT}"

    print("=" * 70)
    print("🚀 VietASR Pro — Mobile Access Mode")
    print("=" * 70)

    # Generate SSL cert
    if not generate_ssl_cert():
        sys.exit(1)

    print()
    print("─" * 70)
    print("📱 HƯỚNG DẪN TRUY CẬP TỪ ĐIỆN THOẠI:")
    print("─" * 70)
    print()
    print("  1️⃣  Đảm bảo điện thoại và máy tính CÙNG mạng WiFi")
    print()
    print(f"  2️⃣  Mở trình duyệt trên điện thoại, truy cập:")
    print(f"     👉  {url_https}")
    print()
    print(f"  3️⃣  Trình duyệt sẽ cảnh báo 'Kết nối không đáng tin cậy'")
    print(f"     → Nhấn 'Nâng cao' / 'Advanced' → 'Tiếp tục' / 'Proceed'")
    print(f"     (Đây là bình thường vì SSL certificate tự ký)")
    print()
    print(f"  4️⃣  Cho phép quyền sử dụng Microphone khi được hỏi")
    print()
    print(f"  🖥️  Truy cập từ máy tính: {url_local}")
    print()

    # QR Code
    print("📱 QR Code (quét bằng điện thoại):")
    print_qr_code(url_https)

    print()
    print("─" * 70)
    print(f"🌐 IP LAN của bạn: {local_ip}")
    print(f"🔒 HTTPS Port: {PORT}")
    print("─" * 70)
    print()

    # Import and run the Flask app with SSL
    sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
    os.chdir(str(Path(__file__).resolve().parent / "app"))

    from app import app, load_model

    print("⏳ Đang tải model AI...")
    model_loaded = load_model()

    if model_loaded:
        print("\n✓ Model đã sẵn sàng!")
    else:
        print("\n⚠ Server chạy không có model (transcription sẽ không hoạt động)")

    print(f"\n🚀 Server đang chạy tại: {url_https}")
    print("   Nhấn Ctrl+C để dừng\n")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(str(CERT_FILE), str(KEY_FILE))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=PORT,
        ssl_context=ssl_context,
    )


if __name__ == "__main__":
    main()
