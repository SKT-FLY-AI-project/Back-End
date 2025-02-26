# pip install tensorflow
# pip install Pillow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image
from app.services.model_loader import cnn_model

## .h5 대용량 모델 로드하기
# 1. 로컬 환경에서 Git LFS가 활성화되었는지 확인
# git lfs install # Git LFS를 전역적으로 설정하고, 로컬 리포지토리에 LFS가 적용되도록 보장해 줘요.
# 2. Git LFS가 .h5 파일을 제대로 추적하고 있는지 확인
# git lfs ls-files
# 3. 로컬에 실제 .h5 파일 다운로드 
# git lfs pull

# 클래스 인덱스 로드 (train_generator.class_indices를 미리 저장해뒀다고 가정)
# train_generator.class_indices를 그대로 사용하거나 저장한 json에서 불러올 수 있음
class_indices = {'A Basket of Clams': 0, 'A Bear Walking': 1, 'A Giant Seated in a Landscape': 2, 'A Goldsmith in his Shop': 3, 'A Gorge in the Mountains (Kauterskill Clove)': 4, 'A Hunting Scene': 5, 'A Rose': 6, 'Adam and Eve': 7, 'Allan Melville': 8, 'Allegory of the Planets and Continents': 9, 'Aman-Jean (Portrait of Edmond François Aman-Jean)': 10, 'Antoine-Laurent Lavoisier (1743–1794) and His Wife': 11, 'Apple Blossoms': 12, 'Approach to a Mountain Village with Horsemen on the Road': 13, 'Approaching Thunder Storm': 14, 'Archangel Gabriel- The Virgin Annunciate': 15, 'Aristotle with a Bust of Homer': 16, 'Arrangement in Flesh Colour and Black': 17, 'At the Circus- The Spanish Walk': 18, 'At the Seaside': 19, 'Autumn Oaks': 20, 'Bacchanal': 21, 'Bacchanal with a Wine Vat': 22, 'Bacchus and Ariadne': 23, 'Bashi-Bazouk': 24, 'Blind Orion Searching for the Rising Sun': 25, 'Boating': 26, 'Boys in a Dory': 27, 'Broken Eggs': 28, 'Burgomaster Jan van Duren (1613–1687)': 29, 'Bust of Pseudo-Seneca': 30, 'Bust of a Man in a Hat Gazing Upward': 31, 'Captain George K. H. Coussmaker (1759–1801)': 32, 'Cardinal Fernando Niño de Guevara (1541–1609)': 33, "Celia Thaxter's Garden, Isles of Shoals, Maine": 34, 'Central Park, Winter': 35, 'Christ Carrying the Cross': 36, 'Christ Carrying the Cross, with the Crucifixion': 37, 'Christ Crucified between the Two Thieves': 38, 'Cider Making': 39, 'Circus Sideshow (Parade de cirque)': 40, 'Clothing the Naked': 41, 'Compositional Sketches for the Virgin Adoring the Christ Child': 42, 'Condesa de Altamira and Her Daughter, María Agustina': 43, 'Coronation of the Virgin': 44, 'Corridor in the Asylum': 45, 'Cottage among Trees': 46, 'Cottage near the Entrance to a Wood': 47, 'Cypresses': 48, 'Daniel Crommelin Verplanck': 49, 'Design for a Wall Monument': 50, 'Diana and Actaeon': 51, 'Diana and Actaeon (Diana Surprised in Her Bath)': 52, 'Dune Landscape with Oak Tree': 53, 'Ecstatic Christ': 54, 'Elijah Boardman': 55, 'Elizabeth Farren': 56, "Emperor Xuanzong's Flight to Shu": 57, 'Erasmus of Rotterdam': 58, 'Ernesta (Child with Nurse)': 59, 'Euphemia White Van Rensselaer': 60, 'Evening Calm, Concarneau, Opus 220 (Allegro Maestoso)': 61, 'Evening Landscape with an Aqueduct': 62, 'Finches and bamboo': 63, 'Fishing Boats, Key West': 64, 'Flight Into Egypt': 65, "Francesco d'Este (born about 1430, died after 1475)": 66, 'Francis Brinley': 67, 'Fur Traders Descending the Missouri': 68, 'Garden at Sainte-Adresse': 69, 'George Washington': 70, 'Great Indian Fruit Bat': 71, 'Head of a Man': 72, 'Head of a Young Woman': 73, 'Heart of the Andes': 74, 'Hercules chasing Avarice from the Temple of the Muses': 75, 'Hermann von Wedigh III (died 1560)': 76, 'Horatio Gates': 77, 'Houses on the Achterzaan': 78, 'Hugh Hall': 79, 'I Saw the Figure 5 in Gold': 80, 'Ia Orana Maria (Hail Mary)': 81, 'Imaginary View of Venice (undivided plate)': 82, 'Improvisation 27 (Garden of Love II)': 83, 'Ink Landscapes with Poems': 84, 'Island of the Dead': 85, 'Joan of Arc': 86, 'Joseph-Antoine Moltedo (born 1775)': 87, 'Juan de Pareja (1606–1670)': 88, 'Jupiter and Juno': 89, 'Lady Lilith': 90, 'Lady at the Tea Table': 91, 'Lady of the Lake': 92, 'Lady with Her Pets (Molly Wales Fobes)': 93, 'Lake George': 94, 'Landscape with Stars': 95, 'Landscape with a Double Spruce in the Foreground': 96, "Landscape—Scene from 'Thanatopsis'": 97, 'Large Boston Public Garden Sketchbook': 98, 'Leisure Time in an Elegant Setting': 99, 'Leonidas at Thermopylae': 100, 'Lucas van Uffel (died 1637)': 101, 'Lucretia': 102, 'Lute Player': 103, 'Madame Félix Gallois': 104, 'Madame Georges Charpentier and Her Children': 105, 'Madame Jacques-Louis Leblanc (Françoise Poncelle, 1788–1839)': 106, 'Madame Roulin and Her Baby': 107, 'Madame X (Madame Pierre Gautreau)': 108, 'Mademoiselle V. . . in the Costume of an Espada': 109, 'Madonna and Child': 110, 'Madonna and Child Enthroned with Saints': 111, 'Manuel Osorio Manrique de Zuñiga (1784–1792)': 112, 'Margaret of Austria': 113, 'Margaretha van Haexbergen (1614–1676)': 114, 'Mars and Venus United by Love': 115, 'Mary Sylvester': 116, 'Maternity': 117, 'May Picture': 118, 'Melencolia I': 119, 'Men Shoveling Chairs (Scupstoel)': 120, 'Merry Company on a Terrace': 121, 'Merrymakers at Shrovetide': 122, 'Mezzetin': 123, 'Michael Angelo and Emma Clara Peale': 124, 'Midshipman Augustus Brine': 125, 'Mme Vuillard Sewing by the Window, rue Truffaut': 126, 'Moonlight on Mount Lafayette, New Hampshire': 127, 'Mountain Stream': 128, 'Mr. and Mrs. Daniel Otis and Child': 129, 'Mrs. Francis Brinley and Her Son Francis': 130, 'Mrs. Grace Dalrymple Elliott (1754–1823)': 131, 'Mrs. Hugh Hammersley': 132, 'Mrs. John Winthrop': 133, 'Mt. Katahdin (Maine), Autumn #2': 134, 'Mäda Primavesi (1903–2000)': 135, 'Night-Shining White': 136, 'Nocturne-The Thames at Battersea': 137, 'Northeaster': 138, 'Note in Pink and Brown': 139, 'Old Plum': 140, 'Old Trees, Level Distance': 141, 'Otto, Count of Nassau and his Wife Adelheid van Vianen': 142, 'Panaromic View of the Bacino di San Marco, Looking up the Giudecca Canal': 143, 'Panoramic View of the Palace and Gardens of Versailles': 144, 'Perseus and the Origin of Coral': 145, 'Piazza San Marco': 146, 'Portrait of Alvise Contarini- (verso) A Tethered Roebuck': 147, 'Portrait of Gerard de Lairesse': 148, 'Portrait of Monsieur Aublet': 149, 'Portrait of Nicolas Trigault in Chinese Costume': 150, "Portrait of Shun'oku Myōha (1311–1388)": 151, 'Portrait of a Carthusian': 152, 'Portrait of a Gentleman': 153, 'Portrait of a German Officer': 154, 'Portrait of a Lady': 155, 'Portrait of a Man, possibly Matteo di Sebastiano di Bernardino Gozzadini': 156, 'Portrait of a Woman with a Man at a Casement': 157, 'Portrait of a Woman, Possibly a Nun of San Secondo- (verso) Scene in Grisaille': 158, "Portrait of a Woman, possibly Ginevra d'Antonio Lupari Gozzadini": 159, 'Portrait of a Young Man': 160, 'Portrait of an Ecclesiastic': 161, 'Portrait of the Artist': 162, 'Portrait of the Painter': 163, 'Princesse de Broglie': 164, 'Prisoners from the Front': 165, 'Queen Esther Approaching the Palace of Ahasuerus': 166, 'Queen Victoria': 167, "Reading the News at the Weavers' Cottage": 168, 'Reclining Female Nude': 169, 'Reclining Nude': 170, 'Road in Etten': 171, 'Rubens, His Wife Helena Fourment (1614–1673), and Their Son Frans (1633–1678)': 172, 'Saada, the Wife of Abraham Ben-Chimol, and Préciada, One of Their Daughters': 173, 'Saint Andrew': 174, 'Saint Anthony the Abbot in the Wilderness': 175, 'Saint Jerome as Scholar': 176, 'Saint John on Patmos': 177, 'Saints Bartholomew and Simon': 178, "Salisbury Cathedral from the Bishop's Grounds": 179, 'Samson Captured by the Philistines': 180, 'Samson and Delilah': 181, 'Satire on Art Criticism': 182, 'Scalinata della Trinità dei Monti': 183, 'Seated Woman': 184, 'Seated Woman, Back View': 185, 'Self-Portrait': 186, 'Self-Portrait, from The Iconography': 187, 'Sibylle': 188, 'Soap Bubbles': 189, 'Spring Blossoms, Montclair, New Jersey': 190, 'Stage Fort across Gloucester Harbor': 191, 'Still Life with Apples and a Pot of Primroses': 192, 'Still Life with Cake': 193, 'Still Life with Jar, Cup, and Apples': 194, 'Still Life- Balsam Apple and Vegetables': 195, 'Street Scene in Paris (Coin de rue à Paris)': 196, 'Studies for the Libyan Sibyl (recto)': 197, "Study for 'Poseuses'": 198, 'Study for the Equestrian Monument to Francesco Sforza': 199, 'Study of a Young Woman': 200, 'Summer Morning': 201, 'Summer Mountains': 202, 'Surf, Isles of Shoals': 203, 'Susan Walker Morse (The Muse)': 204, 'Tahitian Faces (Frontal View and Profiles)': 205, 'Temple Gardens': 206, 'Tennis at Newport': 207, 'The Abduction of Rebecca': 208, 'The Abduction of the Sabine Women': 209, 'The Adoration of the Magi': 210, 'The Adoration of the Shepherds': 211, 'The American School': 212, 'The Annunciation': 213, 'The Annunciation to Zacharias- (verso) The Angel of the Annunciation': 214, 'The Assumption of the Virgin': 215, 'The Beeches': 216, 'The Betrothal of the Virgin': 217, 'The Burial of Punchinello': 218, 'The Card Players': 219, 'The Champion Single Sculls (Max Schmitt in a Single Scull)': 220, 'The Connoisseur': 221, 'The Contest for the Bouquet': 222, 'The Coronation of the Virgin': 223, 'The Creation of the World and the Expulsion from Paradise': 224, 'The Crucifixion': 225, 'The Crucifixion with the Virgin and Saint John': 226, 'The Crucifixion- The Last Judgment': 227, 'The Cup of Tea': 228, 'The Dance Class': 229, 'The Death of Socrates': 230, 'The Denial of Saint Peter': 231, 'The Dining Room': 232, 'The Doorway': 233, 'The Entombment of Christ': 234, 'The Entrance Hall of the Regensburg Synagogue': 235, 'The Falls of Niagara': 236, 'The Feast of Herod and the Beheading of the Baptist': 237, 'The Flower Girl': 238, 'The Fortune-Teller': 239, 'The Genius of Castiglione': 240, 'The Great Statue of Amida Buddha at Kamakura': 241, 'The Gulf Stream': 242, 'The Harvesters': 243, 'The Hatch Family': 244, 'The Head of the Virgin in Three-Quarter View Facing Right': 245, 'The Holy Family with Saints Anne and Catherine of Alexandria': 246, 'The Holy Family with the Young Saint John the Baptist': 247, 'The Horse Fair': 248, 'The Judgment of Paris': 249, "The Kangxi Emperor's Southern Inspection Tour, Scroll Three": 250, 'The Kaunitz Sisters (Leopoldine, Caroline, and Ferdinandine)': 251, 'The Lake of Zug': 252, 'The Last Communion of Saint Jerome': 253, 'The Last Supper': 254, 'The Last Supper, after Leonardo da Vinci': 255, 'The Love Letter': 256, 'The Love Song': 257, 'The Martyrdom of Saint Paul': 258, 'The Meditation on the Passion': 259, 'The Musicians': 260, 'The Nativity': 261, 'The New Bonnet': 262, 'The Parthenon': 263, 'The Penitence of Saint Jerome': 264, 'The Repast of the Lion': 265, 'The Rest on the Flight into Egypt': 266, "The Rocky Mountains, Lander's Peak": 267, "The Round Tower, from 'Carceri d'invenzione'": 268, 'The Sixteen Luohans': 269, 'The Sofa': 270, 'The Sortie Made by the Garrison of Gibraltar': 271, 'The Thinker_ Portrait of Louis N. Kenton': 272, 'The Third-Class Carriage': 273, "The Titan's Goblet": 274, 'The Toilette of Venus': 275, 'The Trojan Women Setting Fire to Their Fleet': 276, 'The Trout Pool': 277, 'The Valley of Wyoming': 278, 'The Virgin of Guadalupe with the Four Apparitions': 279, 'The Young Virgin': 280, 'Three Standing Figures (recto)': 281, 'Three Virtues and Studies of a Seated Man': 282, 'Tommaso di Folco Portinari-Maria Portinari': 283, 'Trees and Houses Near the Jas de Bouffan': 284, 'Twilight on the Sound, Darien, Connecticut': 285, 'Two Men Contemplating the Moon': 286, 'Two Young Girls at the Piano': 287, 'Variations in Violet and Grey—Market Place, Dieppe': 288, 'Venice, from the Porch of Madonna della Salute': 289, 'Venus and Adonis': 290, 'Venus and Cupid': 291, 'View from Mount Holyoke, Northampton, Massachusetts, after a Thunderstorm': 292, 'View of Heidelberg': 293, 'View of Toledo': 294, 'Virgin and Child': 295, 'Virgin and Child with Saint Anne': 296, 'Wang Xizhi watching geese': 297, 'Warwick Castle- The East Front': 298, 'Washington Crossing the Delaware': 299, 'Water-moon Avalokiteshvara': 300, 'Whalers': 301, 'Wheat Field with Cypresses': 302, 'Wheat Fields': 303, 'Winter Scene in Moonlight': 304, 'Wisconsin Landscape': 305, 'Woman Bathing (La Toilette)': 306, 'Woman in a Blue Dress': 307, 'Woman with a Parrot': 308, 'Young Girl Bathing': 309, 'Young Man and Woman in an Inn (Yonker Ramp and His Sweetheart)': 310, 'Young Woman with a Water Pitcher': 311, 'Youth Playing a Pipe for a Satyr': 312, 'Édouard Manet, Seated, Holding His Hat': 313, '꽃다발과 복숭아': 314, '베니스 나의 사랑들': 315, '베니스의 향기로운 모란': 316, '별이 빛나는 밤': 317, '붉은 단풍과 다리': 318, '비아리츠': 319, '스키아 보니의 해안': 320, '아르마운 개양귀비꽃': 321, '여름의 베퇴유': 322, '여름이 다가옵니다': 323, '여름이 다가옵니다': 324, '유채밭': 325, '이즈미르에서의 정박': 326, '장 미빛 인생': 327, '장미와 유리병들': 328, '체리와 꽃': 329, '체리의 계절': 330, '칸과 쉬케': 331, '캐나다의 사과': 332, '코스모스': 333, '파리 인 블루': 334, '파트라시오의 창문': 335, '포도와 꽃다발': 336, '호화로운 꽃다발': 337}
classes = {v: k for k, v in class_indices.items()}

# 이미지 로드 및 전처리
def preprocess_image(image_path, target_size=(224, 224)):
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
async def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = await cnn_model.predict(img_array)
    # print("전체 예측_확률:",prediction)

    max_prob = np.max(prediction)
    threshold = 0.6  # threshold는 기본값 0.7 (70%)
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