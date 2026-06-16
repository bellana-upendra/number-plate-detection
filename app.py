import cv2
import pytesseract

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Correct the file path using double backslashes or a raw string
image = cv2.imread('car1.jpg')

if image is None:
    print("Error: Could not open or find the image.")
else:
    # Convert to Grayscale Image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection
    canny_edge = cv2.Canny(gray_image, 170, 200)

    # Find contours based on edges
    contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    # Initialize license plate contour and coordinates
    license_plate = None
    x, y, w, h = None, None, None, None

    # Find the contour with 4 potential corners and create ROI around it
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4:  # Check if it's a rectangle
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h, x:x + w]
            break

    if license_plate is not None:
        # Thresholding and noise removal
        _, license_plate = cv2.threshold(license_plate, 127, 255, cv2.THRESH_BINARY)
        license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
        _, license_plate = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

        # Text Recognition
        text = pytesseract.image_to_string(license_plate)
        print("License Plate:", text)
    else:
        print("License plate not found.")
