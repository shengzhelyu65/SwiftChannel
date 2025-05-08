from Models.swiftchannel_teacher import SwiftChannelTeacher
from Models.swiftchannel_student import SwiftChannelStudent
from Models.swiftchannel_student_rp import SwiftChannelRP

def return_model(model_selection):
    if model_selection == 'Teacher':
        model = SwiftChannelTeacher(2, 2, upscale=4, feature_channels=24)
    elif model_selection == 'Distillation':
        model = SwiftChannelRP(2, 2, upscale=4, middle_channels=8, feature_channels=12)
    return model