# วิธีแก้ปัญหา "Missing nvrtc DLL"

ปัญหานี้เกิดจากเครื่องคอมพิวเตอร์ของคุณมี Driver การ์ดจอ (สำหรับเล่นเกม) แต่ยังไม่มี **CUDA Toolkit** (สำหรับเขียนโปรแกรม/คำนวณ) ซึ่งจำเป็นสำหรับ CuPy

## วิธีแก้ไข (เลือก 1 ข้อ)

### ทางเลือกที่ 1: ติดตั้ง CUDA Toolkit (แนะนำสำหรับระยะยาว)
เพื่อให้ใช้งาน GPU ได้เต็มประสิทธิภาพ คุณต้องติดตั้ง NVIDIA CUDA Toolkit 12.x

1.  ไปที่เว็บ [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2.  เลือก:
    - **Operating System**: Windows
    - **Architecture**: x86_64
    - **Version**: 10 or 11 (ตาม Windows ของคุณ)
    - **Installer Type**: exe (local)
3.  ดาวน์โหลดและติดตั้ง (ขนาดประมาณ 3GB)
4.  Restart คอมพิวเตอร์
5.  ลองกด "Check GPU" หรือ "Run Benchmark" ในโปรแกรมใหม่อีกครั้ง

### ทดสอบว่าโหลดเสร็จยัง
 - nvcc --version

### ทางเลือกที่ 2: ใช้ CPU Mode ไปก่อน
หากไม่อยากติดตั้งโปรแกรมขนาดใหญ่ตอนนี้ คุณยังสามารถใช้งานโปรแกรมได้โดยเลือกโหมด CPU:

1.  ใน Tab "Spectral Fitting Analysis"
2.  ไปที่ส่วน "Brute-Force Optimization"
3.  เปลี่ยน "Computation Mode" เป็น **CPU (Parallel)**
4.  โปรแกรมจะใช้ CPU หลาย Core ช่วยคำนวณแทน GPU (ช้ากว่าแต่ทำงานได้แน่นอน)

## ทำไมต้องใช้ nvrtc?
Chian (CuPy) ใช้ `nvrtc` (NVIDIA Runtime Compilation) เพื่อแปลงโค้ด Python เป็นภาษาเครื่อง GPU แบบสดๆ (Just-in-Time compilation) เพื่อให้ได้ความเร็วสูงสุดครับ
