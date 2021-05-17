# Project_UZK
Модель автиматизации обработки сейсмоаккустических данных.

Файл preprocessing просто показывает что было сделанно с сырыми данными. все подготовленные данные лежат в папке [datafull_npy.](<https://drive.google.com/drive/folders/1HcyDF5_QoZsIBTfMLAHQARuVtJlrYKC0?usp=sharing>)

inputed_target таргет времени размерность 1х7, velotar таргет скорости размерность 1х6.

Dense_Net_pipeline архитектура модели, генератор данных, обучение и сохранение модели.

accustic.py скрипт позволяет загрузить файл узк получит предсказание модели и посмотреть данные файла.

Predict_from_back использует скрипт accustic.py для загрузки и предсказания.

Presintation.pdf призентация проекта.

[Обученая модел](https://drive.google.com/drive/folders/1_cUDaXwFYPSjiYi8w6lAsP8V6oseprYr?usp=sharing)
