[13:45, 5/26/2026] +90 536 069 22 19: Arkadaslar merhabalar, ödevde etiketlemenin doğru yapılmadığını düsünüyorum bundan dolayı skorlar düsük geliyor. Eğittiğim model degisimleri iyi tespit ediyor ancak true label lar ya eksik ya da hatalı oluyor. Sizlerin sonuçları nasıl geldi?
[13:46, 5/26/2026] +90 536 069 22 19: örnek:
[13:47, 5/26/2026] +90 536 069 22 19: Burada bu degisimlerin hepsi var ama true label da sadece field var
[13:47, 5/26/2026] +90 533 506 39 46: Selamlar, veri setindeki sınıf dengesizliğinden ve veri setindeki veri sayısının azlığından dolayı bende de belli bir sonuçta kaldı aşama 1 ve 2 de ortalama micro f1 0.49 geldi bende
[13:48, 5/26/2026] +90 533 506 39 46: örnek:
Benimkilerde de doğru da bulabiliyor ama zor bulunabilen sınıfları genelde doğru tahmin edemiyor
[13:48, 5/26/2026] +90 555 756 01 61: Bende çok daha düşük skorlar
[13:49, 5/26/2026] +90 536 069 22 19: Selamlar, veri setindeki sınıf dengesizliğinden ve veri setindeki veri sayısının azlığından dolayı bende de belli bir sonuçta kaldı aşama 1 ve 2 de ortalama micro f1 0.49 geldi bende
Micro f1 bende 0.57 ancak macro f1 0.2 lerde
[13:50, 5/26/2026] +90 536 069 22 19: Bende çok daha düşük skorlar
Datasetteki etiketlemenin hatali ve eksik olmasından kaynaklı olduğunu düşünüyorum
[13:50, 5/26/2026] +90 535 685 29 94: Bende de düşük. Veri dengesiz
[13:50, 5/26/2026] +90 536 069 22 19: Attığım örnekte gercekten bina degisimi var mesela ama etiketlenmemis
[13:50, 5/26/2026] +90 536 069 22 19: Bende de düşük. Veri dengesiz
Bir de veri dengesizligi var
[13:51, 5/26/2026] +90 535 685 29 94: 5000 küsür tree varken 20 küsür diğerleri
[13:52, 5/26/2026] +90 535 269 14 02: örnek:
Bu baya yanlış cidden. Sonuçların bu kadar düşük gelmesinin nedeni belli oldu
[13:53, 5/26/2026] +90 535 269 14 02: Micro f1 bende 0.57 ancak macro f1 0.2 lerde
Benim de hemen hemen bu aralıkta çıktı
[13:53, 5/26/2026] +90 553 297 77 10: Benim de makro f1 en fazla 0.6 geldi
[13:54, 5/26/2026] +90 555 756 01 61: Hoca demek ki veri dengesizliğini çözme tekniklerimizi falan görmek istiyor. Ama ben çok da yükseltemedim 🤷🏻‍♀️
[13:54, 5/26/2026] +90 535 269 14 02: Benim de makro f1 en fazla 0.6 geldi
Valla helal olsun. Ben makroda 0.3 üstünü göremedim hiçbir taskta 😄
[13:54, 5/26/2026] +90 533 506 39 46: Bende aynı şekilde
[13:54, 5/26/2026] +90 535 685 29 94: Ben de valla 0.2-0.3 arası zor yükselttim 0.6 baya iyi
[13:55, 5/26/2026] +90 553 297 77 10: Valla helal olsun. Ben makroda 0.3 üstünü göremedim hiçbir taskta 
Şimdi öyle deyince yanlış bir şeyi mi hatırlıyorum emin olamadım 😂
[13:55, 5/26/2026] +90 536 069 22 19: Şimdi öyle deyince yanlış bir şeyi mi hatırlıyorum emin olamadım 
Micro olabilir
[13:55, 5/26/2026] +90 536 069 22 19: Macro ise cok iyi gercekten
[13:55, 5/26/2026] +90 535 269 14 02: Microda object taskinda 0.7leri gördüm mesela. Diğerleri 0.4lerde
[14:09, 5/26/2026] +90 539 565 19 13: Başarılar hep o civarlarda bende de
[14:09, 5/26/2026] +90 539 565 19 13: Model değişmekte fayda etmiyor
[14:09, 5/26/2026] +90 537 046 94 09: Objecte en fazla yüzde 67 gibi diğerleri yuzde 30 yuzde 40 gibi micro f icin
[14:09, 5/26/2026] +90 539 565 19 13: Resnet falan hep aynı metriclerde tıkanıyor 😔
[14:10, 5/26/2026] +90 537 046 94 09: Çıkartamadım daha
[14:11, 5/26/2026] +90 537 046 94 09: Valla helal olsun. Ben makroda 0.3 üstünü göremedim hiçbir taskta 
Bende göremedim max 0.3
[14:17, 5/26/2026] +90 507 844 46 83: Vit kullanan oldumi modelde hic
[14:17, 5/26/2026] +90 507 844 46 83: Ben de resnet kullandim skorlar düşük bende de
[14:18, 5/26/2026] +90 539 565 19 13: Convnext tiny kullandım. Vit cnn karışık birşey
[14:19, 5/26/2026] +90 539 565 19 13: Ama skorlar resnetle hemen hemen aynı
[14:19, 5/26/2026] +90 537 046 94 09: Ben swin kullandım Bende de aynı çok bi fark olmadi
[14:19, 5/26/2026] +90 539 565 19 13: Efficient net kullansam sanki aynı skorlar gelir gibime geliyor.
[14:19, 5/26/2026] +90 537 046 94 09: Resnetle denedim sonra swin le denedim benzer cikti
[14:20, 5/26/2026] +90 537 046 94 09: Bizim veri dengesizliğin çözen bir sey bulmamız lazım
[14:22, 5/26/2026] +90 533 653 80 12: Veri 5 foldda train test’e ayrılıp her foldda test pred’de true label ile pred labeldaki nesne sayısı farkı çok olan örnekler LLM’e atılıp tekrar etiketlenebilir
[14:23, 5/26/2026] +90 533 653 80 12: Anomali yakalayarak bir nevi
[14:23, 5/26/2026] +90 535 269 14 02: Convnext tiny kullandım. Vit cnn karışık birşey
Ben de bunu kullandım. Çok farklı yöntemler denedim ama sorun veri sanki. Yöntem değiştirmek çok ufak etkiler yapıyor
[17:19, 5/26/2026] +90 545 409 25 35: Efficient net kullansam sanki aynı skorlar gelir gibime geliyor.
Ben denedim resnete çok benzer çıkıyor backbone değiştirmek yerine başka şeyler denemek gerekiyor herhalde, ama şuana kadar yaptıklarım da işe çok yaramadı
[17:19, 5/26/2026] +90 533 506 39 46: Vit kullanan oldumi modelde hic
Siyam vit small kullandım
[17:20, 5/26/2026] +90 539 565 19 13: Etiketleri kendi içinde sub fc netlere bölsek nasıl olur acaba
[17:21, 5/26/2026] +90 539 565 19 13: Object labels (unique): ['asphalt', 'building', 'field', 'green', 'land', 'none', 'parking', 'plant', 'road', 'roof', 'tree', 'vegetation', 'water']
Attribute labels (unique): ['adjacent', 'bare', 'black', 'blue', 'brown', 'dark', 'dense', 'empty', 'gray', 'green', 'huge', 'industrial', 'large', 'long', 'lush', 'middle', 'more', 'none', 'paved', 'red', 'residential', 'same', 'small', 'sparse', 'white']
Event labels (unique): ['add', 'appear', 'build', 'change', 'destroy', 'increase', 'none', 'remain', 'remove', 'replace', 'surround', 'turn', 'vegetate']
[17:21, 5/26/2026] +90 539 565 19 13: Etiketlerin bazıları mantıklıds değil.
[17:21, 5/26/2026] +90 539 565 19 13: Middle etiketi var ama neye göre var örnekleri yazdırıyorum yine anlamıyorum 🤣
[17:22, 5/26/2026] +90 539 565 19 13: Siyam vit small kullandım
Sonuç?
[17:37, 5/26/2026] +90 533 506 39 46: micro f1 0.49 larda kaldı macro da 0.19 veri dengesizliği ve çok nadir sınıfları bulamadı model
[20:19, 5/26/2026] +90 534 681 36 32: İyi akşamlar sizin eğitimler uzun mu sürdü?
Özellikle veriyi okurken 
for batch in train_loader:
    img_A,img_B,labels_obj =batch['img_A'].to(device),batch['img_B'].to(device), batch['object_label'].to(device)
Kısım uzun sürüyor sizde de öylemi. Ben veriyi okumak için ayrı bir sınıf kurdum çünkü A ve B çiftleri teker teker ele alınmamalı ve karışmamalılar yani birinin A sı diğerinin B si olmayyacak. Eğitimler yaklaşık ne kadar sürdü? A100 gpu da benim 500k parametreli model her epoch 30 dk gibi bir şey