import cv2
import numpy as np

class Color:
    def __init__(self, lower_bound:list[int], upper_bound: list[int]) -> None:
        self.lower_bound: np.array = np.array(lower_bound, dtype = "uint8")
        self.upper_bound: np.array =  np.array(upper_bound, dtype = "uint8")
    
    def filter_image(self, image_src:np.array) -> np.array:
        #convert the frame to hsv
        frame_hsv = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
        #create a color mask for the cones
        color_mask = cv2.inRange(frame_hsv, self.lower_bound, self.upper_bound)
        color_mask = cv2.erode(color_mask, np.ones((3, 3), np.uint8), iterations = 1)

        #create a new image with a filter
        #frame_filter = cv2.bitwise_or(image_src, image_src, mask = color_mask)
        frame_filter_2 = cv2.bitwise_or(image_src, image_src, mask = color_mask)
        
        #combined = cv2.bitwise_or(frame_filter, frame_filter_2)
        return frame_filter_2

    def get_color_percentage(self, image: np.array) -> float:
        color_mask = cv2.inRange(frame_hsv, self.lower_bound, self.upper_bound)

                # Calculate the total number of pixels in the image
        total_pixels = image.size // image.itemsize

        # Calculate the number of pixels within the specified color range
        color_pixels = np.sum(color_mask > 0)

        # Calculate the percentage of pixels within the specified color range
        color_percentage = (color_pixels / total_pixels) * 100

        return color_percentage


color_blue:Color = Color([106,50,50],[118,255,255])
color_red:Color = Color([0,100,100], [1,255,255])
color_green:Color = Color([51,0,150],[64,100, 255])
frame_hsv: np.array = None

def color_search() -> None:
    #window callback function
    def mouseClick(event, x, y, flags, params):
        #check if buton pressed is a left mouse button being pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            #get the pixel color in hsv from the x an y position of the mouse click and print it
            hsv = frame_hsv[y,x]
            print("The mouse was clicked at x= ", x, "y = ", y)
            print("Hue = ", hsv[0],  "Sat = ", hsv[1], "Val = ", hsv[2])

    cv2.namedWindow("Main")
    cv2.setMouseCallback("Main", mouseClick, param = None)


    # Open the default camera (you can change the parameter to use a different camera)
    cap = cv2.VideoCapture(0)

    # Create a QR code scanner object
    qr_code_scanner = cv2.QRCodeDetector()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        global frame_hsv
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Display the frame
        cv2.imshow('Color isolation red', color_red.filter_image(frame))
        cv2.imshow('Color isolation blue', color_blue.filter_image(frame))
        cv2.imshow('Color isolation green', color_green.filter_image(frame))
        
        # Using cv2.putText() method 
        image = cv2.putText(frame, 'Red: '+str(color_red.get_color_percentage(frame)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (0, 0, 255), 2, cv2.LINE_AA) 
        image = cv2.putText(image, 'Green: '+str(color_green.get_color_percentage(frame)), (50, 200), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (0, 255, 0), 2, cv2.LINE_AA) 
        image = cv2.putText(image, 'Blue: '+str(color_blue.get_color_percentage(frame)), (50, 400), cv2.FONT_HERSHEY_SIMPLEX,  
                        1, (255, 0, 0), 2, cv2.LINE_AA) 
        cv2.imshow('Main', image)
        #print(color_test.get_color_percentage(frame))
        # Check for the 'q' key to exit the loop and terminate the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    qr_data = color_search()
    print("Program terminated.")
