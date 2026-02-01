# Source Clip Details

This directory contains detailed information about the clip used in the project to train our YOLOv8 model to correctly identify motogp bikes on the track. The clips are sections taken from races held in the 2025 MotoGP season.

We will download the source videos (15-20mins). Then extract 1 frame every 5 seconds to insure a diverse set of images for training (approx 120 frames/race). Then filter through for high quality images with clear views of the bikes on the track to create our training dataset (approx 50 images/race).

Download of the source videos is done using yt-dlp, an open-source command-line program to download videos from YouTube and other video platforms. And the extraction of frames is done using OpenCV, a popular computer vision library.

## Shots to KEEP:
| Type of Shot | Description | Why we need it |
| :--- | :--- | :--- |
| **The "Pack"** | 5+ bikes bunched together (Start/Turn 1). Bikes blocking each other. | **Occlusion.** Teaches the model to find individual bikes even when 50% hidden. |
| **The "Knee Dragger"** | Bike leaning >60° in a corner. Rider hanging off. | **Shape.** A leaning bike looks like a wide horizontal blob, distinct from an upright bike. |
| **The "Head-On"** | Bike coming directly at the camera (braking zone). | **Front Profile.** Teaches the narrow, tall silhouette of the fairing/winglets. |
| **The "Side Profile"** | Clear side view on a straight or slight corner. | **Basics.** The clearest view of wheels, fairing, and length. |
| **The "Tiny Dot"** | Helicopter/Drone/Wide shots where bikes are small. | **Scale.** Ensures detection works when the object is only 20x20 pixels. |
| **The "Rain/Wet"** | Riding in rain, spray, or on wet reflective tarmac. | **Noise Filtering.** Prevents detecting reflections on the ground as "ghost" bikes. |
| **The "Rear View"** | Bikes riding away from camera (Grid/Corner Exit). | **Back Profile.** Focuses on rear tires and exhausts. |
| **"Good" Blur** | Blurry background (speed) but bike is relatively distinguishable. | **Reality Check.** TV broadcasts have motion blur; the model must handle it. |

---

## Shots to DELETE:
| Type of Shot | Why Delete? |
| :--- | :--- |
| **Pit Lane / Garage** | **Clutter.** Mechanics, tire warmers, and stands hide the bike's actual shape. |
| **Podium / Parc Fermé** | **Stationary.** Bikes are often covered in flags, or people are standing in front of them. |
| **Onboard Cameras** | **Wrong Perspective.** "Butt cams" or dash views do not look like external bike shots. |
| **Crashes / Gravel** | **Debris.** A bike tumbling through gravel looks like a pile of junk, not a vehicle. |
| **Super Extreme Blur** | **Unrecognizable.** If *you* have to squint to see it, the model won't learn anything. |
| **Crowd / Marshals** | **False Positives.** Shots focused on fans/marshals (unless a bike is clearly visible). |
| **Safety / Medical Car** | **Wrong Class.** We don't want the model to learn that a BMW car is a `bike`. |
| **Obstructed Views** | **Occlusion.** If a massive TV graphic or wall hides >50% of the bike. |

---

## Race Details
For each race, we provide the following details:
- **Race Name**: The official name of the 2025 MotoGP race.
- **Year**: 2025
- **Clip Name**: The filename of the downloaded clip.
- **Source**: URL or platform where the clip was obtained.
- **Duration**: Length of the clip in minutes.
- **Description**: Brief overview of the race conditions (weather, time of day).


Race 1:
- **Race Name**: Qatar GP (Losail)
- **Year**: 2025
- **Clip Name**: 2025_qatar_gp.mp4
- **Source**: https://www.youtube.com/watch?v=tkVp7dcBd6c
- **Duration**: 16:35 (16 minutes, 35 seconds)
- **Description**: Clear, Night Race

Race 2:
- **Race Name**: Spanish GP (Jerez)
- **Year**: 2025
- **Clip Name**: 2025_spanish_gp.mp4
- **Source**: https://www.youtube.com/watch?v=I9fphFbTVz8
- **Duration**: 16:44 (16 minutes, 44 seconds)
- **Description: Sunny, Day Race

Race 3:
- **Race Name**: German GP Sprint (Sachsenring)
- **Year**: 2025
- **Clip Name**: 2025_german_gp_sprint.mp4
- **Source**: https://www.youtube.com/watch?v=k16usQXnvhE
- **Duration**: 4:19 (4 minutes, 19 seconds)
- **Description: Rainy, Day Race

Race 4:
- **Race Name**: Italian GP (Mugello)
- **Year**: 2025
- **Clip Name**: 2025_italian_gp.mp4
- **Source**: https://www.youtube.com/watch?v=R82xflovqx8
- **Duration**: 16:48 (16 minutes, 48 seconds)
- **Description: Sunny, Day Race