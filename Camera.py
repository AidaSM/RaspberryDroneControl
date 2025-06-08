class Camera:
    def __init__(self):
        self.is_on = False
        self.is_recording = False

    def start(self):
        if not self.is_on:
            self.is_on = True
            print("Camera started.")
        else:
            print("Camera is already on.")

    def stop(self):
        if self.is_on:
            if self.is_recording:
                self.stop_recording()
            self.is_on = False
            print("Camera stopped.")
        else:
            print("Camera is already off.")

    def take_photo(self):
        if self.is_on:
            print("Photo taken.")
        else:
            print("Camera is off. Can't take photo.")

    def start_recording(self):
        if self.is_on and not self.is_recording:
            self.is_recording = True
            print("Started video recording.")
        elif not self.is_on:
            print("Camera is off. Can't start recording.")
        else:
            print("Already recording.")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            print("Stopped video recording.")
        else:
            print("Not recording.")
