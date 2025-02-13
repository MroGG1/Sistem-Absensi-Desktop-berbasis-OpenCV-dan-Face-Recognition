import cv2
import face_recognition
import sqlite3
import numpy as np
from tkinter import *
from tkinter import ttk, messagebox
from datetime import datetime
import logging
import openpyxl
import dlib
from scipy.spatial import distance

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Konstanta untuk string yang berulang
DB_PATH = "attendance.db"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\\KK\\shape_predictor_68_face_landmarks.dat")

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Absensi Berbasis Kamera")
        self.root.geometry("500x700")
        self.initialize_db()
        self.create_gui()

    def initialize_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                nim TEXT NOT NULL,
                prodi TEXT NOT NULL,
                face_encoding BLOB)
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL)
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                course_id INTEGER,
                time TEXT,
                status TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (course_id) REFERENCES courses (id))
            ''')
            conn.commit()
        logging.info("Database initialized")

        # Tambahkan kolom prodi jika belum ada
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'prodi' not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN prodi TEXT")
                conn.commit()
                logging.info("Kolom 'prodi' berhasil ditambahkan ke tabel 'users'")

    def load_face_data(self):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, face_encoding FROM users")
                records = cursor.fetchall()

                face_ids = []
                face_names = []
                face_encodings = []

                for record in records:
                    face_ids.append(record[0])
                    face_names.append(record[1])
                    face_encodings.append(np.frombuffer(record[2], dtype=np.float64))

                logging.info("Face data loaded")
                return face_ids, face_names, face_encodings
        except sqlite3.Error as e:
            logging.error(f"An error occurred: {e}")
            return [], [], []

    def record_attendance(self, user_id, course_id, status):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                current_time = datetime.now()
                current_date = current_time.date()

                cursor.execute('''
                SELECT * FROM attendance 
                WHERE user_id = ? AND course_id = ? AND date(time) = ?
                ''', (user_id, course_id, current_date))

                if cursor.fetchone():
                    logging.info("User already recorded attendance today for this course")
                    return

                cursor.execute("INSERT INTO attendance (user_id, course_id, time, status) VALUES (?, ?, ?, ?)",
                               (user_id, course_id, current_time.strftime(DATE_FORMAT), status))
                conn.commit()
                logging.info(f"Attendance recorded with status: {status}")
        except sqlite3.Error as e:
            logging.error(f"An error occurred: {e}")

    def attendance_with_camera(self, course_id):
        face_ids, face_names, known_encodings = self.load_face_data()
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            status = "Tidak Diketahui"

            if not face_encodings:
                cv2.putText(frame, "Wajah tidak terdeteksi", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        user_id = face_ids[best_match_index]
                        user_name = face_names[best_match_index]
                        status = "Present"
                        self.record_attendance(user_id, course_id, status)
                        cv2.putText(frame, f"Absensi Berhasil: {user_name}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        break

            cv2.putText(frame, f"Status: {status}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Absensi Kamera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        logging.info("Camera turned off and returned to main menu")

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def detect_liveness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            ear_left = self.eye_aspect_ratio(left_eye)
            ear_right = self.eye_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2.0

            if ear < 0.2:  # Jika mendeteksi kedipan
                return True
        return False

    def verify_real_face(self, video_capture, known_encodings, known_names):
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            is_real_face = self.detect_liveness(frame)

            if is_real_face:
                for encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, encoding)
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        cv2.putText(frame, f"Welcome {name}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Liveness Detection", frame)
                        cv2.waitKey(2000)  # Tunggu 2 detik
                        return name

            cv2.putText(frame, "Liveness Test Failed", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Liveness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return None

    def select_course_for_attendance(self):
        def proceed_to_attendance():
            selected_course = course_combobox.get()
            if selected_course.strip() == "":
                label_select_status.config(text="Pilih mata kuliah terlebih dahulu!", fg="red")
                return

            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM courses WHERE name = ?", (selected_course,))
                course_id = cursor.fetchone()

            if course_id:
                video_capture = cv2.VideoCapture(0)
                face_ids, face_names, known_encodings = self.load_face_data()
                user_name = self.verify_real_face(video_capture, known_encodings, face_names)
                video_capture.release()
                cv2.destroyAllWindows()

                if user_name:
                    messagebox.showinfo("Liveness Detection", f"Wajah asli terdeteksi. Selamat datang, {user_name}.")
                    self.attendance_with_camera(course_id[0])
                else:
                    messagebox.showwarning("Liveness Detection", "Wajah tidak terdeteksi sebagai wajah asli.")
            else:
                label_select_status.config(text="Mata kuliah tidak ditemukan!", fg="red")
                logging.warning("Course not found")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM courses")
            courses = [row[0] for row in cursor.fetchall()]

        select_window = Toplevel()
        select_window.title("Pilih Mata Kuliah")

        Label(select_window, text="Pilih Mata Kuliah:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        course_combobox = ttk.Combobox(select_window, values=courses, width=30)
        course_combobox.grid(row=0, column=1, padx=5, pady=5)

        Button(select_window, text="Lanjutkan ke Absensi", command=proceed_to_attendance).grid(row=1, columnspan=2, pady=10)
        label_select_status = Label(select_window, text="")
        label_select_status.grid(row=2, columnspan=2, pady=5)

    def register_user_window(self):
        def register_user():
            user_name = entry_user_name.get()
            user_nim = entry_user_nim.get()
            user_prodi = entry_user_prodi.get()

            if user_name.strip() == "" or user_nim.strip() == "" or user_prodi.strip() == "":
                label_user_status.config(text="Semua kolom harus diisi!", fg="red")
                return

            video_capture = cv2.VideoCapture(0)
            label_user_status.config(text="Silakan hadapkan wajah Anda ke kamera...", fg="blue")

            video_capture.read()

            face_encoding = None
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                if face_encodings:
                    face_encoding = face_encodings[0]
                    cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]), (face_locations[0][1], face_locations[0][2]), (0, 255, 0), 2)
                    cv2.putText(frame, "Wajah terdeteksi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                cv2.imshow("Pendaftaran Wajah", frame)

                if face_encoding is not None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    cv2.putText(frame, "Tidak ada wajah terdeteksi, coba lagi...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if face_encoding is not None:
                video_capture.release()
                cv2.destroyAllWindows()

                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO users (name, nim, prodi, face_encoding) VALUES (?, ?, ?, ?)",
                                   (user_name, user_nim, user_prodi, face_encoding.tobytes()))
                    conn.commit()

                label_user_status.config(text="Pengguna berhasil didaftarkan!", fg="green")
                logging.info("User registered successfully")
            else:
                video_capture.release()
                cv2.destroyAllWindows()
                label_user_status.config(text="Gagal mendeteksi wajah. Coba lagi.", fg="red")
                logging.warning("Failed to detect face")

        register_window = Toplevel()
        register_window.title("Tambah Pengguna Baru")

        Label(register_window, text="Nama:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        entry_user_name = Entry(register_window, width=30)
        entry_user_name.grid(row=0, column=1, padx=5, pady=5)

        Label(register_window, text="NIM:").grid(row=1, column=0, padx=5, pady=5, sticky=W)
        entry_user_nim = Entry(register_window, width=30)
        entry_user_nim.grid(row=1, column=1, padx=5, pady=5)

        Label(register_window, text="Prodi:").grid(row=2, column=0, padx=5, pady=5, sticky=W)
        entry_user_prodi = Entry(register_window, width=30)
        entry_user_prodi.grid(row=2, column=1, padx=5, pady=5)

        Button(register_window, text="Daftarkan Pengguna Baru", command=register_user).grid(row=3, columnspan=2, pady=10)
        label_user_status = Label(register_window, text="")
        label_user_status.grid(row=4, columnspan=2, pady=5)

    def add_course_window(self):
        def add_course():
            course_name = entry_course_name.get()
            if course_name.strip() == "":
                label_course_status.config(text="Nama mata kuliah harus diisi!", fg="red")
                return

            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO courses (name) VALUES (?)", (course_name,))
                conn.commit()

            label_course_status.config(text="Mata kuliah berhasil ditambahkan!", fg="green")
            logging.info("Course added successfully")

        course_window = Toplevel()
        course_window.title("Tambah Mata Kuliah")

        Label(course_window, text="Nama Mata Kuliah:").grid(row=0, column=0, padx=5, pady=5, sticky=W)
        entry_course_name = Entry(course_window, width=30)
        entry_course_name.grid(row=0, column=1, padx=5, pady=5)

        Button(course_window, text="Tambahkan Mata Kuliah", command=add_course).grid(row=1, columnspan=2, pady=10)
        label_course_status = Label(course_window, text="")
        label_course_status.grid(row=2, columnspan=2, pady=5)

    def show_attendance(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT users.name, users.nim, users.prodi, courses.name, attendance.time, attendance.status 
            FROM attendance 
            JOIN users ON attendance.user_id = users.id
            JOIN courses ON attendance.course_id = courses.id
            ''')
            records = cursor.fetchall()

        report_window = Toplevel()
        report_window.title("Laporan Absensi")

        cols = ("Nama", "NIM", "Prodi", "Mata Kuliah", "Waktu", "Status")
        tree = ttk.Treeview(report_window, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="center")
        tree.pack(fill=BOTH, expand=True)

        for record in records:
            tree.insert("", "end", values=record)
        logging.info("Attendance report displayed")

    def export_to_xlsx(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
            SELECT users.name, users.nim, users.prodi, courses.name, attendance.time, attendance.status 
            FROM attendance 
            JOIN users ON attendance.user_id = users.id
            JOIN courses ON attendance.course_id = courses.id
            ''')
            rows = cursor.fetchall()

            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "Laporan Absensi"

            headers = ["Nama", "NIM", "Prodi", "Mata Kuliah", "Waktu", "Status"]
            sheet.append(headers)

            for row in rows:
                sheet.append(row)

            workbook.save("Laporan_Absensi.xlsx")
            logging.info("Laporan berhasil diekspor ke Laporan_Absensi.xlsx")
            messagebox.showinfo("Sukses", "Laporan berhasil diekspor ke Laporan_Absensi.xlsx")
        except Exception as e:
            logging.error(f"Error saat mengekspor laporan: {e}")
            messagebox.showerror("Error", f"Error saat mengekspor laporan: {e}")
        finally:
            conn.close()

    def create_frame(self, text, button_text, command):
        frame = LabelFrame(self.root, text=text, padx=10, pady=10)
        frame.pack(padx=10, pady=10, fill="x")
        Button(frame, text=button_text, command=command).pack(pady=10)
        return frame

    def create_gui(self):
        self.create_frame("Absensi Kamera", "Pilih Mata Kuliah untuk Absensi", self.select_course_for_attendance)
        self.create_frame("Laporan Absensi", "Lihat Laporan", self.show_attendance)
        self.create_frame("Manajemen Pengguna", "Tambah Pengguna Baru", self.register_user_window)
        self.create_frame("Manajemen Mata Kuliah", "Tambah Mata Kuliah Baru", self.add_course_window)
        self.create_frame("Ekspor Laporan", "Ekspor ke XLSX", self.export_to_xlsx)

if __name__ == "__main__":
    root = Tk()
    app = AttendanceSystem(root)
    root.mainloop()