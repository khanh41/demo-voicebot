class Download:
    download_url = {
        "scrfd_face_detect": [
            "/home/khanhpluto/khanhtv/sharingan/face-recognition/app/resources/scrfd_2.5g_bnkps_shape640x640.onnx",
            "https://www.dropbox.com/s/4l7mq1q8u2y7lwr/scrfd_2.5g_bnkps_shape640x640.onnx?dl=1"],
        "face_recognition": ["app/resources/recognition.clf",
                             "https://www.dropbox.com/s/rp7geuz1vjnsmcy/recognition.clf?dl=1"],
        "face_recognition_mask": ["app/resources/recognition_mask.clf",
                                  'https://www.dropbox.com/s/k88m0vgcl82e26z/recognition_mask.clf?dl=1'],
    }


class Chatbot:
    age_classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                   '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028',
                   '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042',
                   '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056',
                   '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070',
                   '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084',
                   '085', '086', '087', '088', '089', '090', '091', '092', '093', '095', '096', '099', '100', '101',
                   '103', '105', '110', '111', '115', '116']
    age_messages = ["Do you believe I can guess your age?", "Do you want me to guess your age?",
                    "let me guess your age"]

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_messages = {
        "angry": "You seem angry. Who makes you angry? Tell me I'll punch it",
        "disgust": "You seem angry. Who makes you angry? Tell me I'll punch it",
        "fear": "Your face doesn't look good. What are you worried about?",
        "happy": "You seem so happy? What are you happy about today? Can you tell me?",
        "neutral": "You seem so happy? What are you happy about today? Can you tell me?",
        "sad": "Your face doesn't look good. What are you sad about?",
        "surprise": "You seem so happy? What are you happy about today? Can you tell me?",
    }
