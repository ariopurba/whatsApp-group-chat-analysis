## Ekstrak Data

Pada bagian ekstrak data kita terlebih dahulu dapat mengambilnya dari chat group whatsapp kita. Untuk lebih jelas bagaimana mengeksport data chat group dapat dilihat pada [Cara menyimpan riwayat chat Anda
](https://faq.whatsapp.com)

<br>

## Import Library dan Loading Data
<br>
Pertama kita mengimport library yang akan digunakan dalam memproses data chat group. Saya akan menggunakan library sastrawi untuk memproses data chat teks dikarenakan data yang saya gunakan merupakan chat group whatsapp keluarga saya yang menggunakan bahasa indonesia. Teman- teman juga dapat menggunakan library nlp lain seperti NLTK untuk teks berbasa inggris.

``` python
# Import Library
import emoji
import numpy as np
import pandas as pd
import re
import regex  
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud

import matplotlib.pyplot as plt
%matplotlib inline

```
Setelah itu kita akan mencoba melihat data dan menganalisisnya, untuk code dari preprocessing yang saya gunakan diambil dari [article ini](https://thecleverprogrammer.com/2020/08/06/whatsapp-group-chat-analysis/).

berikut merupakan kode preprocesing yang saya gunakan.
``` python
# ekstrak data tanggal
def date_time(s):
  pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
  result = re.match(pattern, s)
  if result:
    return True
  return False

# extract username in the chats group
def find_author(s):
  s = s.split(":")
  if len(s)==2:
    return True
  else:
    return False

# Separate all information
def GetDataPoint(line):
  splitLine = line.split(' - ')
  dateTime = splitLine[0]
  date, time = dateTime.split(', ')
  message = ' '.join(splitLine[1:])
  if find_author(message):
    splitMessage = message.split(': ')
    author = splitMessage[0]
    message = ' '.join(splitMessage[1:])
  else:
    author = None
  return date, time, author, message

# extract emoji 
def split_count(text):

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI['en'] for char in word):
            emoji_list.append(word)

    return emoji_list

```
kemudian kita akan mencoba untuk menggunakan function-function di atas pada kode berikut.

```python

parseData= []
data = []
text_data = '/content/purba_wa.txt'
with open(text_data, encoding = "utf-8") as fp:
  fp.readline()
  messageBuffer = []
  date, time, partisipan = None, None, None
  while True:
    line = fp.readline()
    if not line:
      break
    line = line.strip()
    if date_time(line):
      if len(messageBuffer) > 0:
        parseData.append([date, time, partisipan, ' '.join(messageBuffer)]) 
      messageBuffer.clear()
      date, time, partisipan, message = GetDataPoint(line)
      messageBuffer.append(message)
    else:
      messageBuffer.append(line)

```

Lalu saya akan membuat dataframe dari data yang sudah kita ekstrak dimana memiliki kolom Tanggal, waktu, Nama User, Pesan, Emoji, Panjang Pesan.

```python
# Buat data frame
clean_data = pd.DataFrame(parseData, columns=['Tanggal',
                                              'Waktu',
                                              'Nama User',
                                              'Pesan'])#make dataframe

clean_data['Tanggal'] = pd.to_datetime(clean_data['Tanggal'])#Buat format tanggal

clean_data['Emoji'] = clean_data['Pesan'].apply(split_count)#ekstrak emoji dari pesan

# kolom panjang pesan
panjang = [len(clean_data['Pesan'][i]) for i in range(len(clean_data))]
clean_data['Panjang Pesan'] = panjang

clean_data # lihat data
```

![img](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*itfyN-JpIbKW-vX5ON2Q3w.png)

Shape dari data
Sebelum kita menganalisis lebih lanjut data yang sudah di ekstrak, kita perlu untuk membersihkan atau mengcleaning data yang tidak kita perlu seperti pesan “<Media omitted>” dan “This message was deleted”. Kita perlu untuk membersihkannya untuk keperluan analisis lebih lanjut.
```python
# Bersihkan Data Pesan
clean_data = clean_data[clean_data.Pesan != '<Media omitted>']
clean_data = clean_data[clean_data.Pesan != 'This message was deleted']
```
## Menganalisis Data

Pada bagian ini banyak hal yang dapat kita lakukan, seperti mevisualisakan kata yang sering digunakan dan banyak lagi.

Menampilkan Kata Yang Sering Digunakan Dalam WordCloud
Terlebih dahulu kita memproses datanya.
```python
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

factory = StopWordRemoverFactory()
stopwords = factory.create_stop_word_remover()

messages = []

for words in clean_data['Pesan']:
    only_letters = re.sub("[^a-zA-Z]", " ", words)
    tokens = only_letters.split()  # tokenize the sentences
    lower_case = [l.lower() for l in tokens]  # convert all letters to lower case
    filtered_result = [word for word in lower_case if not stopwords.remove(word)] # Remove stopwords from the comments
    lemmas = [stemmer.stem(t) for t in filtered_result]  # lemmatizes the words to their base form

    messages.append(' '.join(lemmas))
```

setelah itu kita dapat menampilkan dalam wordcloud.

```python
#Let's use worldcloud to visualize the messages
unique_string=(" ").join(messages)
wordcloud = WordCloud(width = 2000, height = 1000,background_color='white').generate(unique_string)
plt.figure(figsize=(20,12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![image](https://miro.medium.com/v2/resize:fit:786/format:webp/1*ngFskkzQxXjJEq61ZmRNog.png)

Dapat terlihat kata yang sering digunakan oleh group whatsapp saya.

## Melihat Emoji Yang Sering Digunakan
Dari data yang kita punya, kita juga dapat melihat apa saja emoji yang sering digunakan dalam mengirim pesan chat.

```python
# Cek Emoji Paling Sering Digunakan
total_emojis_list = list([a for b in clean_data.Emoji for a in b])
emoji_dict = dict(Counter(total_emojis_list))
emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
for i in emoji_dict:
  print(i)

```

## Mengecek Jumlah Pesan Yang Dikirim Oleh setiap user
Untuk melihat user yang paling aktif di sebuah group kita dapat mengeceknya.

```python
# Cek Pengirim pesan terbanyak
total_chat = clean_data['Nama User'].tolist()
user_dict = dict(Counter(total_chat))
user_dict = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)
for i in user_dict:
  print(i)
```
![image](https://miro.medium.com/v2/resize:fit:568/format:webp/1*XUhHc8xOvYm0_OBmumIWbw.png)

## Kesimpulan

Banyak hal yang bisa kita dapatkan dari menganalisis pesan teks whatsapp group seperti emoji yang paling sering digunakan ketika berkirim pesan, user dengan traffic mengirim pesan terbanyak, dan masih banyak lagi.

Saya mengetahui bahwa yang saya kerjakan ini masih banyak kekurangan, namun saya akan tetap mencoba memperbaikinya lagi dan mencoba menjadi lebih baik setiap harinya. Terima Kasih Sudah membaca. Yoo Semangat Belajar Coding…..
