# Hybrid Steganographic Protocol

This repository demonstrates how to run the **Hybrid Steganographic Protocol** using two Ubuntu virtual machines (VMs), simulating real-world distributed communication between **Sender** (the sender) and **Recipient** (the receiver). The protocol combines elements of **steganography by cover synthesis** and **steganography by cover modification** to enhance security, ensuring covert and unsuspicious communication.

---

## 📚 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Cloning the Repository](#1️⃣-clone-the-repository)
3. [Installing Required Packages](#2️⃣-install-required-python-packages)
4. [Configuring Virtual Machines](#3️⃣-setup-virtual-machines-for-sender-amara-and-receiver-ebere)
5. [Running the Servers](#5️⃣-run-servers-on-eberes-vm-receiver)
6. [Sending and Embedding Messages](#7️⃣-send-m_1-and-m_2-to-ebere)
7. [Extracting the Hidden Message](#🔄-10️⃣-verify-and-decode-the-message-on-eberes-vm)

---

## 📋 **Prerequisites**

To run this protocol, you will need:

- **Ubuntu** (20.04, 22.04, or any supported version) on two separate virtual machines.
- **Python 3.8 or higher** installed on both VMs.
- **Git** to clone the repository.
- **Pip3** to manage Python packages.
- **Required Python libraries**: Flask, NumPy, Requests, and Pillow.

---

## 1️⃣ **Clone the Repository**

On both **Sender's VM** and **Recipient's VM**, open a terminal and run:

```bash
# Clone the repository
git clone <YOUR_GITHUB_REPO_URL> hybrid-steganographic-protocol
cd hybrid-steganographic-protocol
```
> **Note:** Replace `<YOUR_GITHUB_REPO_URL>` with the URL of your GitHub repository.

---

## 2️⃣ **Install Required Python Packages**

Run the following commands on **both VMs**:

```bash
sudo apt update
sudo apt install python3-pip python3-venv -y

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```
> **Note:** If `requirements.txt` is not provided, you can install required libraries manually:
```bash
pip install flask numpy requests pillow
```

---

## 3️⃣ **Setup Virtual Machines for Sender (Sender) and Receiver (Recipient)**

### **On Sender's VM (Sender)**
- Scripts to be run on Sender's VM:
  - `setup_and_synth.py`
  - `send_m1_m2.py`
  - `lsb_embed.py`
  - `server_1.py` (for \(m_1\))
  - `server_2.py` (for \(m_2\))

### **On Recipient's VM (Receiver)**
- Scripts to be run on Recipient's VM:
  - `server_1.py` (for \(m_1\))
  - `server_2.py` (for \(m_2\))
  - `server_3.py` (for stego-object \(s\) and MAC)
  - `lsb_decode.py` (for decoding and verifying the message)

---

## 4️⃣ **Configure IP Addresses**

1. Find the IP address of both VMs using:
   ```bash
   ip a
   ```

2. Update the following scripts on Sender's VM with Recipient's IP address:
   
   **`send_m1_m2.py`**
   ```python
   server_1_url = "http://192.168.56.103:5001/send_m1"  # Update with Recipient's IP
   server_2_url = "http://192.168.56.103:5002/send_m2"  # Update with Recipient's IP
   ```

   **`lsb_embed.py`**
   ```python
   server_url = "http://192.168.56.103:5003/receive_stego"  # Update with Recipient's IP
   ```

---

## 5️⃣ **Run Servers on Recipient's VM (Receiver)**

On **Recipient's VM**, run the following commands in separate terminal tabs:

```bash
# Start server to receive m_1 (Port 5001)
python3 server_1.py

# Start server to receive m_2 (Port 5002)
python3 server_2.py

# Start server to receive stego-object and MAC (Port 5003)
python3 server_3.py
```

---

## 6️⃣ **Run Scripts on Sender's VM (Sender)**

On **Sender's VM**, run the following commands:

1. **Run Setup and Synthesis**
```bash
python3 setup_and_synth.py
```
This generates the following files:
- `m1.txt`
- `m2.txt`
- `Masked Message (Binary).txt`

2. **Send \(m_1\) and \(m_2\) to Recipient**
```bash
python3 send_m1_m2.py
```

3. **Embed the secret into the Stego-Image**
```bash
python3 lsb_embed.py
```

4. **Transmit Stego-Image and MAC to Recipient**
The script will automatically transmit the stego-image and MAC to **Recipient's server (Port 5003)**.

---

## 🔄 **10️⃣ Verify and Decode the Message on Recipient's VM**

After receiving the stego-image and MAC, Recipient verifies the MAC and extracts the secret message.

On **Recipient's VM**, run:

```bash
python3 lsb_decode.py
```
This script will:
1. Extract the masked message \(b\) from the stego-image.
2. Unmask the original secret message \(m\).

---

## 📂 **Directory Structure**
```
hybrid-steganographic-protocol/
  ├── setup_and_synth.py
  ├── send_m1_m2.py
  ├── lsb_embed.py
  ├── lsb_decode.py
  ├── server_1.py
  ├── server_2.py
  ├── server_3.py
  └── data/
      ├── m1.txt
      ├── m2.txt
      └── Masked Message (Binary).txt
```

---

## 🔥 **Congratulations!**
You have successfully implemented and run the **Hybrid Steganographic Protocol** on two Ubuntu VMs. For any issues or contributions, please submit an issue or pull request on the GitHub repository.

**Happy Steganography!**

