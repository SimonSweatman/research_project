**Cloning the repository into VS Code**

Using the VS Code interface

1. Open the GitHub repository in a browser.
2. Select Code → HTTPS and copy the repository URL.
3. Open VS Code.
4. Press Ctrl+Shift+P and select Git: Clone.
5. Paste the copied GitHub URL and choose where to save the repository.
6. Select Open when cloning finishes.
7. If prompted, select Trust the authors of this folder.
The PlatformIO project is inside the 6012 research_project folder. Open this folder directly in VS Code using:
File → Open Folder → 6012 research_project

Using the terminal

Alternatively, run:
1. git clone YOUR_GITHUB_REPOSITORY_URL
2. cd YOUR_REPOSITORY_NAME
3. code "6012 research_project"
4. Replace YOUR_GITHUB_REPOSITORY_URL with the HTTPS link copied from GitHub.


Required VS Code extension
1. Install the PlatformIO IDE extension from the VS Code Extensions menu. Once the project folder is open, PlatformIO should detect platformio.ini and install the required Arduino Uno platform and PCA9685 library automatically.
2. The firmware can then be compiled and uploaded using the Build and Upload buttons on the PlatformIO toolbar. Connect the Arduino and use PlatformIO: Serial Monitor at 115200 baud to control the array.



**Using the path designer**
Requirements
- Python 3 with Tkinter support
- Matplotlib or Pillow is recommended for exporting path-preview images
- PlatformIO with the Arduino Uno platform

From the repository, open a terminal and move into the PlatformIO project folder:
1. cd "6012 research_project"
2. python scripts/cilia_path_designer.py

Creating a trajectory
1. Move the paddle tip by dragging it on the canvas or using the X/Y-position and servo-angle controls.
2. Select Save coordinate to add individual points, or use Start live trace to record a continuous path.
3. Use Undo, Clear, the arc tool and corner fillets to refine the trajectory.
4. Ensure that the final point returns close to the starting point to prevent a sudden movement when the path repeats.
5. Set a cycle duration and select Play to check the motion of the two-link paddle.

Checking the complete array
1. Open the Array Simulator tab and select Use current path.
2. Set the number of cilia, pivot spacing and adjacent phase shift.
3. Select Play to inspect the coordinated array motion.
4. Select Check complete cycle to check all 360 simulated positions for collisions. Do not operate the physical array if a collision or clearance warning is shown.
5. The phase–spacing safety map can be used to compare a wider range of possible settings.

Exporting the path
1. Select Arduino PWM header from the export menu.
2. Set the required number of lookup samples. The experimental paths used 360 samples with Resample uniformly along path enabled.
3. Select Export path and save the file as:
include/gait_table.h

This header contains the calibrated raw PWM commands used by the Arduino controller. Exporting a new path will replace the existing gait table, so retain a copy of any path that may be required again.




**Running the physical cilia array**

Uploading the controller
1. Check that the servo connections, PWM boards and external 5 V supplies match the completed array wiring. The servo power must not be supplied from the Arduino USB connection.
2. Ensure that the paddles can move freely and that no object or obstruction is within the array during start-up.
3. Open the 6012 research_project folder in PlatformIO.
4. Connect the Arduino Uno by USB and upload src/main.cpp.
5. Apply power to the servo supplies with the motion area clear. On start-up, all cilia move to their calibrated 90 degrees home position and remain stopped.

Serial Monitor control

Open the PlatformIO Serial Monitor at 115200 baud and select a newline or carriage-return line ending. Type help to display the available commands.

Command	Action
1. phase 180	Set the adjacent phase shift to 180 degrees. Valid range: 0–360°. The array must be stopped.
2. speed 3	Set the complete cycle duration to 3 s. Valid range: 0.5–120 s.
3. start	Move smoothly into the gait and begin continuous operation. Also resumes from a pause.
4. pause	Hold the cilia at their current positions.
5. stop	Stop the gait and return all cilia smoothly to 90 degrees.
6. home	Same action as stop.
7. status	Display the current state, cycle duration and phase shift.
8. help	Display the command list.

Here, speed refers to the cycle duration in seconds; a smaller value produces faster operation.

Example:
- phase 180
- speed 3
- status
- start
- Enter stop before changing the phase shift or placing hands near the array. The red synchronisation LED connected to pin 7 briefly illuminates at the start of each completed cycle.
