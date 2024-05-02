# wtech_yapay_zeka

# MNIST Veri Kümesiyle Yapay Zeka Modeli

Bu proje, Python programlama dili kullanılarak MNIST veri kümesi üzerinde bir yapay zeka modeli oluşturmayı amaçlamaktadır. MNIST veri kümesi, el yazısı rakamları içeren bir veri kümesidir ve genellikle derin öğrenme modellerinin başlangıç noktası olarak kullanılır.

## Proje İçeriği
  - Veri Seti Seçimi: MNIST veri kümesi seçildi.
  - Veri Seti Hazırlama ve Ön İşleme: Veri seti hazırlandı, eksik veriler dolduruldu, kategorik veriler sayısal verilere dönüştürüldü ve gereksiz veriler çıkarıldı.
  - Model Seçimi ve Eğitimi: Evrişimli Sinir Ağı (CNN) modeli seçildi ve eğitildi.
  - Model içi FastAPI ile API Oluşturma: Eğitilen model, FastAPI kullanılarak bir API haline getirildi.
  - Sunum: Projeyi ve elde edilen sonuçları anlatan bir sunum hazırlandı.



## Kullanım

1. **Gereksinimler**

   - Python 3.x
   - Keras
   - TensorFlow
   - NumPy
   - Matplotlib
   - Jupyter Notebook

2. Kurulum

   Gereksinimleri yüklemek için aşağıdaki komutları kullanın:

   ```bash
   pip install numpy matplotlib tensorflow keras jupyter
   ```

3. Veri Kümesi

   Mnist veri kümesi, derin öğrenme modeli eğitimi ve testi için kullanılacaktır. Veri kümesi otomatik olarak yüklenir.

4. Eğitim

   Modeli eğitmek için `CNN_model_egitim.ipynb` Jupyter Notebook dosyasını kullanın.

   ```bash
   jupyter notebook CNN_model_egitim.ipynb
   ```

   Notebook içindeki talimatları takip ederek modeli eğitebilirsiniz.


Aşağıda, FastAPI kullanarak bir görüntü sınıflandırma uygulaması için bir "readme" şablonu bulunmaktadır:

---

# Görüntü Sınıflandırma Uygulaması

Bu proje, FastAPI kullanarak bir görüntü sınıflandırma servisi sunar. Kullanıcılar, uygulamaya bir görüntü yükleyerek, bu görüntünün hangi sınıfa ait olduğunu tahmin edebilirler.

## Kurulum

1. **Gereksinimler**

   - Python 3.x
   - FastAPI
   - Uvicorn
   - Python-Multipart
   - TensorFlow
   - OpenCV
   - NumPy

   Gereksinimleri yüklemek için aşağıdaki komutları kullanın:

   ```bash
   pip install -r requirements.txt
   ```

2. **Model ve Servis Hazırlığı**

   - Görüntü sınıflandırma modeli ve ağırlıkları (`CNN_six_layer_fully_connected.json` ve `CNN_six_fully_connected.h5`) bulunmalıdır. Bu dosyaların, bu proje dizininde olduğundan emin olun.
   - Servis, görüntüyü işlemek ve tahminler yapmak için OpenCV ve NumPy kütüphanelerini kullanır. Gerekirse bu kütüphaneleri yükleyin.

3. **Uygulamayı Başlatma**

   Uygulamayı başlatmak için aşağıdaki komutu kullanın:

   ```bash
   uvicorn app:app --reload
   ```

   Bu komut, `app.py` dosyasındaki FastAPI uygulamasını başlatır ve değişiklikler otomatik olarak algılanarak yeniden yüklenir.

4. **Kullanım**

   - Uygulama başlatıldıktan sonra, bir web tarayıcısında `http://localhost:8000` adresine gidin.
   - Sayfada, bir görüntü dosyası seçin ve "Yükle" düğmesine basın.
   - Servis, yüklenen görüntüyü alacak ve sınıflandıracak ve sonucu ekranda gösterecektir.

## Notlar
- Bu uygulama, önceden eğitilmiş bir görüntü sınıflandırma modelini kullanır. Modeli eğitmek ve yeni verilere göre güncellemek istiyorsanız, ayrıntılı talimatlar için model belgelerine bakın.
- FastAPI hizmetine bir görüntü yüklemek için basit bir HTML formu bulunmaktadır. Bu form, kullanıcıların bir görüntü seçmelerine ve sunucuya yüklemelerine olanak tanır. Yüklenen görüntü, FastAPI hizmeti tarafından tahmin edilir ve sonuç JSON olarak sunulur. Ayrıca, Mnist veri seti ve CNN modeli hakkında bilgilendirici bir bölüm de içerir.
  
