import cv2
from pyzbar.pyzbar import decode

def scan_qr_code():
    # Open the default camera (you can change the parameter to use a different camera)
    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    # Create a QR code scanner object
    qr_code_scanner: cv2.QRCodeDetector = cv2.QRCodeDetector()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Detect QR codes in the frame
        retval, decoded_info, points, straight_qrcode = qr_code_scanner.detectAndDecodeMulti(frame)

        # Check if QR code is detected
        if retval:
            # Loop through detected QR codes
            for info in decoded_info:
                # Print the QR code data
                print(f"QR Code Data: {info}")

            # Release the camera and terminate the program
            #cap.release()
            #cv2.destroyAllWindows()
            #return decoded_info

        # Display the frame
        cv2.imshow('QR Code Scanner', frame)

        # Check for the 'q' key to exit the loop and terminate the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    qr_data = scan_qr_code()
    print("Program terminated.")
