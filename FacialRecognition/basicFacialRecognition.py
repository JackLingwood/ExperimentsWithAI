import os
import sys


# https://pypi.org/project/face-recognition/


sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole

clearConsole()

# Set current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())


import face_recognition


def samePerson(image1Filename, image2Filename):
    known_image = face_recognition.load_image_file(image1Filename)
    unknown_image = face_recognition.load_image_file(image2Filename)

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    return results[0]

def report(imageFilename1, imageFilename2):
    base1, ext1 = os.path.splitext(imageFilename1)
    base2, ext2 = os.path.splitext(imageFilename1)

    heading2("Same person?", base1 + " vs " + base2 + " = " + str(samePerson(imageFilename1, imageFilename2)))

images = [
    "JimCarrey1.jpg",
    "JimCarrey2.jpg",
    "JimCarrey3.jpg",
    "JimCarrey4.jpg",
    "JimCarrey5.jpg",
    "GeorgeClooney1.jpg",
    "GeorgeClooney2.jpg",
    "GeorgeClooney3.jpg",
    "GeorgeClooney4.jpg",
    "GeorgeClooney5.jpg"]

heading("Basic Facial Recognition")

print("Comparing two images to see if they are the same person...")

def addSubfolder(imageFilename, subfolder=".\\InputImages"):
    base, ext = os.path.splitext(imageFilename)
    folder = os.path.join(os.getcwd(), subfolder)
    new_filename = os.path.join(folder, os.path.basename(base) + ext)
    return new_filename


for image in images:
    for image2 in images:
        if image != image2:
            report(addSubfolder(image), addSubfolder(image2))



def getFilenameWithExtension(filename):
    base, ext = os.path.splitext(filename)
    if ext == "":
        ext = ".jpg"  # Default to .jpg if no extension is found
    return base + ext

def getFaceEncodings(imageFilename):
    image = face_recognition.load_image_file(imageFilename)
    face_encodings = face_recognition.face_encodings(image)
    heading2("Face Encodings", face_encodings)
    if (face_encodings):
        print(f"Found {len(face_encodings)} face encodings in the image.")
        return face_encodings[0]
    else:
        print("No face encodings found in the image.")

import json

def getFaceLandmarks(imageFilename):
    face_encoding = getFaceEncodings(imageFilename)
    encoding_list = face_encoding.tolist()
    encoding_json = json.dumps(encoding_list)
    heading2("Face Encoding JSON -" + imageFilename, encoding_json)
    return encoding_json

for image in images:
    getFaceLandmarks(addSubfolder(image))

heading("Final section: Face Encodings and Landmarks")

image = face_recognition.load_image_file(addSubfolder("JimCarrey1.jpg"))
face_locations = face_recognition.face_locations(image)
print(f"Found {len(face_locations)} face(s) in the image.")
print(face_locations)

face_landmarks_list = face_recognition.face_landmarks(image)
heading2("Face Landmarks", face_landmarks_list)