import cv2, dlib, PIL.Image, argparse, os, ntpath
import numpy as np
from imutils import face_utils
from pathlib import Path


class recognition:
    
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description='Welcome in Recognition App')
        self.parser.add_argument('-i', '--input', type=str, required=True)

        print(f'Starting ...')
        print(f'Importing pretrained models ...')
        
        self.main()


    def main(self):

        self.load()
        args = self.parser.parse_args()

        print('Importing faces ...')
        face_to_encode_path = Path(args.input)
        files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

        for file_ in face_to_encode_path.rglob('*.png'):
            files.append(file_)
        if len(files)==0:
            raise ValueError('No faces has been detected {}'.format(face_to_encode_path))
        known_face_names = [os.path.splitext(ntpath.basename(file_))[0] for file_ in files]

        known_face_encodings = []
        for file_ in files:
            image = PIL.Image.open(file_)
            image = np.array(image)
            face_encoded = self.encode_face(image)[0][0]
            known_face_encodings.append(face_encoded)

        print('Faces imported ...')
        print('Starting Webcam...')
        video_capture = cv2.VideoCapture(0)
        print('Webcam started ...')
        print('Trying to detect ...')
        
        while True:
            ret, frame = video_capture.read()
            self.face_reco(frame, known_face_encodings, known_face_names)
            cv2.imshow('Recognition App', frame)
        print('System has been stopped')
        video_capture.release()
        cv2.destroyAllWindows()
    
    def load():

        self.pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
        self.pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
        self.face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
        self.face_detector = dlib.get_frontal_face_detector()

    def transform(self, image, face_locations):
        
        coord_faces = []
        for face in face_locations:
            rect = face.top(), face.right(), face.bottom(), face.left()
            coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
            coord_faces.append(coord_face)
        return coord_faces


    def encode_face(self, image):
        
        face_locations = self.face_detector(image, 1)
        face_encodings_list = []
        landmarks_list = []
        
        for face_location in face_locations:
            
            shape = self.pose_predictor_68_point(image, face_location)
            face_encodings_list.append(np.array(self.face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
            
            shape = face_utils.shape_to_np(shape)
            landmarks_list.append(shape)
        
        face_locations = self.transform(image, face_locations)
        return face_encodings_list, face_locations, landmarks_list


    def face_reco(self, frame, known_face_encodings, known_face_names):
        
        rgb_small_frame = frame[:, :, ::-1]
       
        face_encodings_list, face_locations_list, landmarks_list = self.encode_face(rgb_small_frame)
        face_names = []
        for face_encoding in face_encodings_list:
            if len(face_encoding) == 0:
                return np.empty((0))
       
            vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
            tolerance = 0.6
            result = []
            for vector in vectors:
                if vector <= tolerance:
                    result.append(True)
                else:
                    result.append(False)
            if True in result:
                first_match_index = result.index(True)
                name = known_face_names[first_match_index]
            else:
                name = "Unknown"
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations_list, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        for shape in landmarks_list:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)



if __name__ == '__main__':
    recognition()