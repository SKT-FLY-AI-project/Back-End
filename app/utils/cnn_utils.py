# pip install tensorflow
# pip install Pillow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# 모델 로드
model = tf.keras.models.load_model(r"app\utils\cnn_model_250217_1.h5")


## .h5 대용량 모델 로드하기
# 1. 로컬 환경에서 Git LFS가 활성화되었는지 확인
# git lfs install # Git LFS를 전역적으로 설정하고, 로컬 리포지토리에 LFS가 적용되도록 보장해 줘요.
# 2. Git LFS가 .h5 파일을 제대로 추적하고 있는지 확인
# git lfs ls-files
# 3. 로컬에 실제 .h5 파일 다운로드 
# git lfs pull


    
# 클래스 인덱스 로드 (train_generator.class_indices를 미리 저장해뒀다고 가정)
# train_generator.class_indices를 그대로 사용하거나 저장한 json에서 불러올 수 있음
class_indices = {'A_Basket_of_Clams': 0, 'A_Bear_Walking': 1, 'A_Giant_Seated_in_a_Landscape__sometimes_called__T': 2, 'A_Goldsmith_in_his_Shop': 3, 'A_Gorge_in_the_Mountains__Kauterskill_Clove_': 4, 'A_Hunting_Scene': 5, 'A_Rose': 6, 'Adam_and_Eve': 7, 'Allan_Melville': 8, 'Allegory_of_the_Planets_and_Continents': 9, 'Aman_Jean__Portrait_of_Edmond_Fran_ois_Aman_Jean_': 10, 'Antoine_Laurent_Lavoisier__1743_1794__and_His_Wife': 11, 'Apple_Blossoms': 12, 'Approach_to_a_Mountain_Village_with_Horsemen_on_th': 13, 'Approaching_Thunder_Storm': 14, 'Archangel_Gabriel__The_Virgin_Annunciate': 15, 'Aristotle_with_a_Bust_of_Homer': 16, 'Arrangement_in_Flesh_Colour_and_Black__Portrait_of': 17, 'At_the_Circus__The_Spanish_Walk__Au_Cirque__Le_Pas': 18, 'At_the_Seaside': 19, 'Autumn_Oaks': 20, 'Bacchanal': 21, 'Bacchanal_with_a_Wine_Vat': 22, 'Bacchus_and_Ariadne': 23, 'Bashi_Bazouk': 24, 'Blind_Orion_Searching_for_the_Rising_Sun': 25, 'Boating': 26, 'Boys_in_a_Dory': 27, 'Broken_Eggs': 28, 'Burgomaster_Jan_van_Duren__1613_1687_': 29, 'Bust_of_Pseudo_Seneca': 30, 'Bust_of_a_Man_in_a_Hat_Gazing_Upward': 31, 'Captain_George_K__H__Coussmaker__1759_1801_': 32, 'Cardinal_Fernando_Ni_o_de_Guevara__1541_1609_': 33, 'Celia_Thaxter_s_Garden__Isles_of_Shoals__Maine': 34, 'Central_Park__Winter': 35, 'Christ_Carrying_the_Cross': 36, 'Christ_Carrying_the_Cross__with_the_Crucifixion__T': 37, 'Christ_Crucified_between_the_Two_Thieves__The_Thre': 38, 'Cider_Making': 39, 'Circus_Sideshow__Parade_de_cirque_': 40, 'Clothing_the_Naked': 41, 'Compositional_Sketches_for_the_Virgin_Adoring_the_': 42, 'Condesa_de_Altamira_and_Her_Daughter__Mar_a_Agusti': 43, 'Coronation_of_the_Virgin': 44, 'Corridor_in_the_Asylum': 45, 'Cottage_among_Trees': 46, 'Cottage_near_the_Entrance_to_a_Wood': 47, 'Cypresses': 48, 'Daniel_Crommelin_Verplanck': 49, 'Design_for_a_Wall_Monument': 50, 'Diana_and_Actaeon': 51, 'Diana_and_Actaeon__Diana_Surprised_in_Her_Bath_': 52, 'Dune_Landscape_with_Oak_Tree': 53, 'Ecstatic_Christ': 54, 'Elijah_Boardman': 55, 'Elizabeth_Farren__born_about_1759__died_1829___Lat': 56, 'Erasmus_of_Rotterdam': 57, 'Ernesta__Child_with_Nurse_': 58, 'Euphemia_White_Van_Rensselaer': 59, 'Evening_Calm__Concarneau__Opus_220__Allegro_Maesto': 60, 'Evening__Landscape_with_an_Aqueduct': 61, 'Fishing_Boats__Key_West': 62, 'Flight_Into_Egypt': 63, 'Francesco_d_Este__born_about_1430__died_after_1475': 64, 'Francis_Brinley': 65, 'Fur_Traders_Descending_the_Missouri': 66, 'Garden_at_Sainte_Adresse': 67, 'George_Washington': 68, 'Great_Indian_Fruit_Bat': 69, 'Head_of_a_Man': 70, 'Head_of_a_Young_Woman': 71, 'Heart_of_the_Andes': 72, 'Hercules_chasing_Avarice_from_the_Temple_of_the_Mu': 73, 'Hermann_von_Wedigh_III__died_1560_': 74, 'Horatio_Gates': 75, 'Houses_on_the_Achterzaan': 76, 'Hugh_Hall': 77, 'I_Saw_the_Figure_5_in_Gold': 78, 'Ia_Orana_Maria__Hail_Mary_': 79, 'Imaginary_View_of_Venice__undivided_plate_': 80, 'Improvisation_27__Garden_of_Love_II_': 81, 'Island_of_the_Dead': 82, 'Joan_of_Arc': 83, 'Jos_phine__l_onore_Marie_Pauline_de_Galard_de_Bras': 84, 'Joseph_Antoine_Moltedo__born_1775_': 85, 'Juan_de_Pareja__1606_1670_': 86, 'Jupiter_and_Juno__Study_for_the__Furti_di_Giove__T': 87, 'Lady_Lilith': 88, 'Lady_at_the_Tea_Table': 89, 'Lady_of_the_Lake': 90, 'Lady_with_Her_Pets__Molly_Wales_Fobes_': 91, 'Lake_George': 92, 'Landscape_Scene_from__Thanatopsis_': 93, 'Landscape_with_Stars': 94, 'Landscape_with_a_Double_Spruce_in_the_Foreground': 95, 'Large_Boston_Public_Garden_Sketchbook__The_Hunting': 96, 'Leisure_Time_in_an_Elegant_Setting': 97, 'Leonidas_at_Thermopylae': 98, 'Lucas_van_Uffel__died_1637_': 99, 'Lucretia': 100, 'Lute_Player': 101, 'M_da_Primavesi__1903_2000_': 102, 'Madame_F_lix_Gallois': 103, 'Madame_Georges_Charpentier__Margu_rite_Louise_Lemo': 104, 'Madame_Jacques_Louis_Leblanc__Fran_oise_Poncelle__': 105, 'Madame_Roulin_and_Her_Baby': 106, 'Madame_X__Madame_Pierre_Gautreau_': 107, 'Mademoiselle_V______in_the_Costume_of_an_Espada': 108, 'Madonna_and_Child': 109, 'Madonna_and_Child_Enthroned_with_Saints': 110, 'Manuel_Osorio_Manrique_de_Zu_iga__1784_1792_': 111, 'Margaret_of_Austria': 112, 'Margaretha_van_Haexbergen__1614_1676_': 113, 'Mars_and_Venus_United_by_Love': 114, 'Mary_Sylvester': 115, 'Maternity': 116, 'May_Picture': 117, 'Melencolia_I': 118, 'Men_Shoveling_Chairs__Scupstoel_': 119, 'Merry_Company_on_a_Terrace': 120, 'Merrymakers_at_Shrovetide': 121, 'Mezzetin': 122, 'Michael_Angelo_and_Emma_Clara_Peale': 123, 'Midshipman_Augustus_Brine': 124, 'Mme_Vuillard_Sewing_by_the_Window__rue_Truffaut': 125, 'Moonlight_on_Mount_Lafayette__New_Hampshire': 126, 'Mountain_Stream': 127, 'Mr__and_Mrs__Daniel_Otis_and_Child': 128, 'Mrs__Francis_Brinley_and_Her_Son_Francis': 129, 'Mrs__Grace_Dalrymple_Elliott__1754__1823_': 130, 'Mrs__Hugh_Hammersley': 131, 'Mrs__John_Winthrop': 132, 'Mt__Katahdin__Maine___Autumn__2': 133, 'Nocturne__Nocturne__The_Thames_at_Battersea_': 134, 'Northeaster': 135, 'Note_in_Pink_and_Brown': 136, 'Otto__Count_of_Nassau_and_his_Wife_Adelheid_van_Vi': 137, 'Panaromic_View_of_the_Bacino_di_San_Marco__Looking': 138, 'Panoramic_View_of_the_Palace_and_Gardens_of_Versai': 139, 'Perseus_and_the_Origin_of_Coral': 140, 'Piazza_San_Marco': 141, 'Portrait_of_Alvise_Contarini______verso__A_Tethere': 142, 'Portrait_of_Gerard_de_Lairesse': 143, 'Portrait_of_Monsieur_Aublet': 144, 'Portrait_of_Nicolas_Trigault_in_Chinese_Costume': 145, 'Portrait_of_a_Carthusian': 146, 'Portrait_of_a_Gentleman': 147, 'Portrait_of_a_German_Officer': 148, 'Portrait_of_a_Lady': 149, 'Portrait_of_a_Man__possibly_Matteo_di_Sebastiano_d': 150, 'Portrait_of_a_Woman__Possibly_a_Nun_of_San_Secondo': 151, 'Portrait_of_a_Woman__possibly_Ginevra_d_Antonio_Lu': 152, 'Portrait_of_a_Woman_with_a_Man_at_a_Casement': 153, 'Portrait_of_a_Young_Man': 154, 'Portrait_of_an_Ecclesiastic': 155, 'Portrait_of_the_Artist': 156, 'Portrait_of_the_Painter': 157, 'Prisoners_from_the_Front': 158, 'Queen_Esther_Approaching_the_Palace_of_Ahasuerus': 159, 'Queen_Victoria': 160, 'Reading_the_News_at_the_Weavers__Cottage': 161, 'Reclining_Female_Nude': 162, 'Reclining_Nude': 163, 'Road_in_Etten': 164, 'Rubens__His_Wife_Helena_Fourment__1614_1673___and_': 165, 'Saada__the_Wife_of_Abraham_Ben_Chimol__and_Pr_ciad': 166, 'Saint_Andrew': 167, 'Saint_Anthony_the_Abbot_in_the_Wilderness': 168, 'Saint_Jerome_as_Scholar': 169, 'Saint_John_on_Patmos': 170, 'Saints_Bartholomew_and_Simon': 171, 'Salisbury_Cathedral_from_the_Bishop_s_Grounds': 172, 'Samson_Captured_by_the_Philistines': 173, 'Samson_and_Delilah': 174, 'Satire_on_Art_Criticism': 175, 'Scalinata_della_Trinit__dei_Monti': 176, 'Seated_Woman': 177, 'Seated_Woman__Back_View': 178, 'Self_Portrait': 179, 'Self_Portrait__from_The_Iconography': 180, 'Sibylle': 181, 'Soap_Bubbles': 182, 'Spring_Blossoms__Montclair__New_Jersey': 183, 'Stage_Fort_across_Gloucester_Harbor': 184, 'Still_Life__Balsam_Apple_and_Vegetables': 185, 'Still_Life_with_Apples_and_a_Pot_of_Primroses': 186, 'Still_Life_with_Cake': 187, 'Still_Life_with_Jar__Cup__and_Apples': 188, 'Street_Scene_in_Paris__Coin_de_rue___Paris_': 189, 'Studies_for_the_Libyan_Sibyl__recto___Studies_for_': 190, 'Study_for__Poseuses_': 191, 'Study_for_the_Equestrian_Monument_to_Francesco_Sfo': 192, 'Study_of_a_Young_Woman': 193, 'Summer_Morning': 194, 'Surf__Isles_of_Shoals': 195, 'Susan_Walker_Morse__The_Muse_': 196, 'Tahitian_Faces__Frontal_View_and_Profiles_': 197, 'Temple_Gardens': 198, 'Tennis_at_Newport': 199, 'The_Abduction_of_Rebecca': 200, 'The_Abduction_of_the_Sabine_Women': 201, 'The_Adoration_of_the_Magi': 202, 'The_Adoration_of_the_Shepherds': 203, 'The_American_School': 204, 'The_Annunciation': 205, 'The_Annunciation_to_Zacharias___verso__The_Angel_o': 206, 'The_Assumption_of_the_Virgin': 207, 'The_Beeches': 208, 'The_Betrothal_of_the_Virgin': 209, 'The_Burial_of_Punchinello': 210, 'The_Card_Players': 211, 'The_Champion_Single_Sculls__Max_Schmitt_in_a_Singl': 212, 'The_Connoisseur': 213, 'The_Contest_for_the_Bouquet__The_Family_of_Robert_': 214, 'The_Coronation_of_the_Virgin': 215, 'The_Creation_of_the_World_and_the_Expulsion_from_P': 216, 'The_Crucifixion': 217, 'The_Crucifixion__The_Last_Judgment': 218, 'The_Crucifixion_with_the_Virgin_and_Saint_John': 219, 'The_Cup_of_Tea': 220, 'The_Dance_Class': 221, 'The_Death_of_Socrates': 222, 'The_Denial_of_Saint_Peter': 223, 'The_Dining_Room': 224, 'The_Doorway': 225, 'The_Entombment_of_Christ': 226, 'The_Entrance_Hall_of_the_Regensburg_Synagogue': 227, 'The_Falls_of_Niagara': 228, 'The_Feast_of_Herod_and_the_Beheading_of_the_Baptis': 229, 'The_Flower_Girl': 230, 'The_Fortune_Teller': 231, 'The_Genius_of_Castiglione': 232, 'The_Great_Statue_of_Amida_Buddha_at_Kamakura__Know': 233, 'The_Gulf_Stream': 234, 'The_Harvesters': 235, 'The_Hatch_Family': 236, 'The_Head_of_the_Virgin_in_Three_Quarter_View_Facin': 237, 'The_Holy_Family_with_Saints_Anne_and_Catherine_of_': 238, 'The_Holy_Family_with_the_Young_Saint_John_the_Bapt': 239, 'The_Horse_Fair': 240, 'The_Judgment_of_Paris': 241, 'The_Judgment_of_Paris__he_is_sitting_at_left_with_': 242, 'The_Kaunitz_Sisters__Leopoldine__Caroline__and_Fer': 243, 'The_Lake_of_Zug': 244, 'The_Last_Communion_of_Saint_Jerome': 245, 'The_Last_Supper': 246, 'The_Last_Supper__after_Leonardo_da_Vinci': 247, 'The_Love_Letter': 248, 'The_Love_Song': 249, 'The_Martyrdom_of_Saint_Paul': 250, 'The_Meditation_on_the_Passion': 251, 'The_Musicians': 252, 'The_Nativity': 253, 'The_New_Bonnet': 254, 'The_Parthenon': 255, 'The_Penitence_of_Saint_Jerome': 256, 'The_Repast_of_the_Lion': 257, 'The_Rest_on_the_Flight_into_Egypt': 258, 'The_Rocky_Mountains__Lander_s_Peak': 259, 'The_Round_Tower__from__Carceri_d_invenzione___Imag': 260, 'The_Sofa': 261, 'The_Sortie_Made_by_the_Garrison_of_Gibraltar': 262, 'The_Thinker__Portrait_of_Louis_N__Kenton': 263, 'The_Third_Class_Carriage': 264, 'The_Titan_s_Goblet': 265, 'The_Toilette_of_Venus': 266, 'The_Trojan_Women_Setting_Fire_to_Their_Fleet': 267, 'The_Trout_Pool': 268, 'The_Valley_of_Wyoming': 269, 'The_Virgin_of_Guadalupe_with_the_Four_Apparitions': 270, 'The_Young_Virgin': 271, 'Three_Standing_Figures__recto___Seated_Woman_and_a': 272, 'Three_Virtues__Temperance__Hope__and_Fortitude_or_': 273, 'Tommaso_di_Folco_Portinari__1428_1501___Maria_Port': 274, 'Trees_and_Houses_Near_the_Jas_de_Bouffan': 275, 'Twilight_on_the_Sound__Darien__Connecticut': 276, 'Two_Men_Contemplating_the_Moon': 277, 'Two_Young_Girls_at_the_Piano': 278, 'Variations_in_Violet_and_Grey_Market_Place__Dieppe': 279, 'Venice__from_the_Porch_of_Madonna_della_Salute': 280, 'Venus_and_Adonis': 281, 'Venus_and_Cupid': 282, 'View_from_Mount_Holyoke__Northampton__Massachusett': 283, 'View_of_Heidelberg': 284, 'View_of_Toledo': 285, 'Virgin_and_Child': 286, 'Virgin_and_Child_with_Saint_Anne': 287, 'Warwick_Castle__The_East_Front': 288, 'Washington_Crossing_the_Delaware': 289, 'Whalers': 290, 'Wheat_Field_with_Cypresses': 291, 'Wheat_Fields': 292, 'Winter_Scene_in_Moonlight': 293, 'Wisconsin_Landscape': 294, 'Woman_Bathing__La_Toilette_': 295, 'Woman_in_a_Blue_Dress': 296, 'Woman_with_a_Parrot': 297, 'Young_Girl_Bathing': 298, 'Young_Man_and_Woman_in_an_Inn___Yonker_Ramp_and_Hi': 299, 'Young_Woman_with_a_Water_Pitcher': 300, 'Youth_Playing_a_Pipe_for_a_Satyr': 301, '____Buddha_of_Medicine_Bhaishajyaguru__Yaoshi_fo_': 302, '_____Old_Plum': 303, '_________Illustrated_Legends_of_the_Kitano_Tenjin_': 304, '_________Portrait_of_Shun_oku_My_ha__1311_1388_': 305, '____________Night_Shining_White': 306, '_____________Summer_Mountains': 307, '______________Emperor_Xuanzong_s_Flight_to_Shu': 308, '________________The_Sixteen_Luohans': 309, '__________________Finches_and_bamboo': 310, '____________________Wang_Xizhi_watching_geese': 311, '______________________Ink_Landscapes_with_Poems': 312, '_________________________Old_Trees__Level_Distance': 313, '__________________________________The_Kangxi_Emper': 314, '_douard_Manet__Seated__Holding_His_Hat': 315, '수월관음도_고려__________Water_moon_Avalokiteshvara': 316}

classes = {v: k for k, v in class_indices.items()}  # 숫자 인덱스를 클래스명으로 변환

# 테스트할 이미지 경로
# test_image_path = r"app\models\two_cnn\data\cnn_test_data\test_ViewofToledo_ElGreco.png" # Confidence: 99.98%
# test_image_path = r"app\models\two_cnn\data\cnn_test_data\test_GardenatSainte-Adresse_monet.png"  # Confidence: 100.00%

# 이미지 로드 및 전처리
def preprocess_image(image_path, target_size=(150, 150)):
    # `image_path`가 numpy 배열이면 변환
    if isinstance(image_path, np.ndarray):
        img = Image.fromarray(image_path)  # numpy 배열을 PIL 이미지로 변환
    elif isinstance(image_path, str):
        img = load_img(image_path)  # 이미지 크기 조정
    else:
        raise TypeError("image_path must be a file path (str) or numpy.ndarray.")
    
    img = img.resize(target_size)  # 이미지 크기 조정
    img_array = img_to_array(img) / 255.0  # 0~1 범위로 정규화
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3) 형태로 변환
    return img_array

# 예측 수행
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    # print("전체 예측_확률:",prediction)

    max_prob = np.max(prediction)
    threshold = 0.9  # threshold는 기본값 0.7 (70%)
    if max_prob < threshold:
        print("Unclassified: 이미지가 학습된 클래스에 속하지 않습니다.")

        predicted_class = np.argmax(prediction)  # 가장 높은 확률의 클래스 인덱스
        class_name = classes[predicted_class]
        class_name = classes[predicted_class]
        confidence = prediction[0][predicted_class] * 100
        print(f"참고로, 가장 높은 Confidence({confidence:.2f}% )를 가진 작품은 {class_name} 입니다. ")

        return "Unknown Title"
    
    else:
        predicted_class = np.argmax(prediction)  # 가장 높은 확률의 클래스 인덱스

        class_name = classes[predicted_class]
        confidence = prediction[0][predicted_class] * 100
        print(f"Predicted class: {class_name}, Confidence: {confidence:.2f}%")

        return {class_name}

# # 테스트 이미지 예측
# if os.path.exists(test_image_path):
#     predict_image(test_image_path)
# else:
#     print(f"이미지 파일을 찾을 수 없습니다: {test_image_path}")