import os
import json
import numpy as np 
import pandas as pd 

def read_json(file_name):
    with open(file_name) as f:
        json_data =json.loads(f.read())
    return json_data

def prepare_data_by_file(file_name):
    json_data = read_json(file_name)
    df = pd.DataFrame(json_data["messages"])
    df["text_bold"] = df["text"].apply(get_text_bold_from_dict)
    df["text_hashtag"] = df["text"].apply(get_text_hashtag_from_dict)
    
    return df

def get_text_bold_from_dict(data):
    res = data[0] if data else ""
    for el in data:
        if el and isinstance(el, dict) and el.get("type") == "bold":
            res =el.get("text")
    return res

def get_text_hashtag_from_dict(data):
    res = ""
    for el in data:
        if el and isinstance(el, dict) and el.get("type") == "hashtag":
            res =el.get("text")
    return res

dfs = []
for dirname, _, filenames in os.walk('data/input/archive'):
    for filename in filenames:
        file_name = os.path.join(dirname, filename)
        try:
            dfs.append(prepare_data_by_file(file_name))
        except Exception as error:
            print(file_name)
            print(error)
            print("===============================================")

df_all = pd.concat(dfs, axis=0)

df_all.head()

df = df_all[['id', 'date', 'text_bold', 'text_hashtag']]

def get_alert_flag(data):
    if "🔴" in data:
        return "Start"
    if "🟢" in data:
        return "Stop"
    if "🟡" in data:
        return "Stop partial"
    return np.nan

df["alert_flag"] = df.text_bold.map(get_alert_flag)
df["text_hashtag"] = df["text_hashtag"].str.replace("#", "")

df = df[~df.alert_flag.isna()]

translations = {'Уманська_територіальна_громада': 'Uman territorial community', 'Софіївська_територіальна_громада': 'Sofia territorial community', 'Липецька_територіальна_громада': 'Lipetsk territorial community', 'м_Старокостянтинів_та_Старокостянтинівська_територіальна_громада': 'M_T -steamer_that_starokostiantynivska_ territorial', 'Чугуївський_район': 'Chuguev district', 'Братська_територіальна_громада': 'Saratov territorial community', 'Красноградський_район': 'Krasnograd district', 'Ніжинський_район': 'Nizhyn_ region', 'Київська_область': 'Kiev_Bost', 'Володимирівська_територіальна_громада': 'Vladimir territorial community', 'м_Ватутіне_та_Ватутінська_територіальна_громада': 'm_vatutine_thatutin_ territorial', 'Гребінківська_територіальна_громада': 'Grebinkivsky territorial community', 'Купянський_район': 'Kupyansky_ region', 'Кальміуський_район': 'Kalmius district', 'Бахмутський_район': 'Bakhmut district', 'Сумська_область': 'Sumy_blast', 'Василівська_територіальна_громада': 'Vasyliv Territorial Community', 'Бердянський_район': 'Berdyansk_ district', 'Мирівська_територіальна_громада': 'Mirovskaya Territorial Community', 'Чернівецька_область': 'Chernivtsi region', 'м_Бориспіль_та_Бориспільська_територіальна_громада': 'm Boryspil and_ Boryspil Territorial Community', 'м_Краматорськ_та_Краматорська_територіальна_громада': 'M_ Kramatorsk_T_TA_COMATATIVE_THERORIORIORIAL COMMUNICATION', 'м_Житомир_та_Житомирська_територіальна_громада': 'm Zhytomyr _T_ Zhytomyr Territorial Community', 'Закарпатська_область': 'Transcarpathian region', 'Ізюмський_район': 'Izium_ region', 'Полтавська_область': 'Poltava region', 'Прибужанівська_територіальна_громада': 'Pribuzhanivsky territorial community', 'Малотокмачанська_територіальна_громада': 'Malostmachan territorial community', 'Черкаська_територіальна_громада': 'Cherkasy territorial community', 'Чернівецький_район': 'Chernivtsi_ region', 'м_Шостка_та_Шосткинська_територіальна_громада': 'M_Sostka_The Shatskinskaya_Thetotorial', 'Тестовий_Регіон': 'Test region', 'м_Лубни_та_Лубенська_територіальна_громада': 'm_lubny_t_lubenskaya_ territorial', 'Прилуцький_район': 'Prylutsky_ district', 'м_Кремінна_та_Кремінська_територіальна_громада': 'M_ Kremin_that_rekminka_Thertotorical Community', 'Житомирський_район': 'Zhytomyr_ region', 'Синюхинобрідська_територіальна_громада': 'Sinyukhinobrida territorial community', 'Харківська_область': 'Kharkiv region', 'Вараський_район': 'Varas district', 'Херсон': 'Kherson', 'Хмільницький_район': 'Khmelnytskyi district', 'м_Синельникове_та_Синельниківська_територіальна_громада': 'M_sinelnikov_that_ynelnikovskaya_ territorial', 'Вугледарська_територіальна_громада': 'The coal territorial community', 'Дубровицька_територіальна_громада': 'Dubrovytska territorial community', 'Березанська_територіальна_громада': 'Berezan territorial community', 'Гуляйпільська_територіальна_громада': 'Gulyaypil territorial community', 'Веснянська_територіальна_громада': 'Desnyansk territorial community', 'Гостомелська_територіальна_громада': 'Gostomelskaya_ territorial', 'Луганська_область': 'Lugansk region', 'Сарненський_район': 'Sarny district', 'МогилівПодільський_район': 'Mogilev Podilsky_', 'Херсонська_область': 'Kherson_ex', 'Олешківська_територіальна_громада': 'Oleshkiv Territorial Community', 'Черкаська_область': 'Cherkasy region', 'Горохівська_територіальна_громада': 'Gorokhiv territorial community', 'Воскресенська_територіальна_громада': 'Resurrection territorial community', 'м_Вознесенськ_та_Вознесенська_територіальна_громада': 'M_Voznesensk_Th_Voznesenskaya_ territorial', 'Врадіївська_територіальна_громада': 'Brody territorial community', 'Дорошівська_територіальна_громада': 'Toporovsky territorial community', 'м_Снігурівка_та_Снігурівська_територіальна_громада': 'm Snihurivka and_ Snihurivska Territorial Community', 'Куцурубська_територіальна_громада': 'Kutsuruba territorial community', 'Охтирський_район': 'Okhtyrsky district', 'Очеретинська_територіальна_громада': 'Ochychin territorial community', 'Пісочинська_територіальна_громада': 'Vysochansk territorial community', 'Чернігівський_район': 'Chernihiv_ region', 'Вознесенський_район': 'Voznesensky district', 'м_Рівне_та_Рівненська_територіальна_громада': 'm Rivne_t_t_ Rivne territorial community', 'Вінницька_область': 'Vinnytsia region', 'м_Кременчук_та_Кременчуцька_територіальна_громада': 'M_ Kremenchuk_t_ Kremenchuk_Therteritorial_Con', 'Циркунівська_територіальна_громада': 'Tsirkunivsky Territorial Community', 'Дергачівська_територіальна_громада': 'Dergachiv territorial community', 'м_Обухів_та_Обухівська_територіальна_громада': 'm_bukhiv_ta_obukhivska_T territorial', 'Ізмаїльський_район': 'Izmail district', 'Кропивницький_район': 'Kropyvnytskyi district', 'Малинівська_територіальна_громада': 'Kalinovsky territorial community', 'Львівська_область': 'Lviv_Bost', 'Благодатненська_територіальна_громада': 'The Blessed Territorial Community', 'Мигіївська_територіальна_громада': 'MIIIA Territorial Community', 'м_Дніпро_та_Дніпровська_територіальна_громада': 'm Dnipro and Dniprovsk Territorial Community', 'м_Черкаси_та_Черкаська_територіальна_громада': 'm Cherkasy _T_ Cherkasy Territorial Community', 'Звенигородський_район': 'Zvenigorod district', 'Первомайська_територіальна_громада': 'Pervomaisk territorial community', 'Великоновосілківська_територіальна_громада': 'Velikovosilkivsky Territorial Community', 'м_НоваОдеса_та_Новоодеська_територіальна_громада': 'M_NOVESSA_TA_NOVODESSKA_THERORIORIORIORIORIAL COMMUNICATION', 'Подільський_район': 'Podilskyi district', 'Сватівський_район': 'Swativ district', 'Золочівська_територіальна_громада': 'Zolochiv territorial community', 'Черкаський_район': 'Cherkasy_ region', 'Вінницький_район': 'Vinnytsia_ region', 'Степівська_територіальна_громада': 'Stepiv Territorial Community', 'м_Баштанка_та_Баштанська_територіальна_громада': 'M_Bashtanka_ta_Bashtanskaya_ territorial', 'Коростенський_район': 'Korosten district', 'Золотоніський_район': 'Zolotonsky district', 'м_Лисичанськ_та_Лисичанська_територіальна_громада': 'M_lisichansk_T_lisichanskaya_The community', 'м_Сарни_та_Сарненська_територіальна_громада': 'M_Sarny_tay_sarnenska_Thertotorial', 'м_Южноукраїнськ_та_Южноукраїнська_територіальна_громада': 'M_uzhnoukrainsk_Thu.uzhnoukrainskaya_The community', 'Коблівська_територіальна_громада': 'Kobliv Territorial Community', 'Смілянська_територіальна_громада': 'Malinskaya Territorial Community', 'Харківський_район': 'Kharkiv_ district', 'Дружківська_територіальна_громада': 'Rudkivsky Territorial Community', 'м_Біла_Церква_та_Білоцерківська_територіальна_громада': 'M_Bila_Cerkva_ta_Bilotserkivska_Thertotorial', 'м_Київ': 'M_Kiiv', 'Тульчинський_район': 'Tulchyn district', 'Роздільнянський_район': 'Razdelnyansky district', 'м_Марганець_та_Марганецька_територіальна_громада': 'M_Marganets_t_Marganetska_Thertotorial', 'Мостівська_територіальна_громада': 'Mostiv Territorial Community', 'Баштанський_район': 'Bashtansky district', 'Донецька_область': 'Donetsk region', 'Миронівська_територіальна_громада': 'Mironovsky territorial community', 'Уманський_район': 'Uman_ region', 'Бахмутська_територіальна_громада': 'Bakhmut territorial community', 'Запорізький_район': 'Zaporizhzhya district', 'Мелітопольський_район': 'Melitopol district', 'Новомарївська_територіальна_громада': 'Novomaryiv territorial community', 'Болградський_район': 'Bolgrad_ region', 'Українська_територіальна_громада': 'Ukrainian territorial community', 'Криворізький_район': 'Kryvyi Rih district', 'Пологівський_район': 'Pologovsky district', 'Макарівська_територіальна_громада': 'Makariv territorial community', 'Маріупольський_район': 'Mariupol district', 'м_Васильків_та_Васильківська_територіальна_громада': 'm Vasylkiv and_ Vasylkivska Territorial Community', 'Єланецька_територіальна_громада': 'Yelanets territorial community', 'м_Павлоград_та_Павлоградська_територіальна_громада': 'm_pavlograd_ta_pavlograd_ territorial_ community', 'Шепетівський_район': 'Shepetivskyi district', 'Гірська_територіальна_громада': 'Mountain territorial community', 'Святогірська_територіальна_громада': 'Svyatogorsk Territorial Community', 'ІваноФранківська_область': 'Ivano Frankivsk region', 'Запорізька_область': 'Zaporizhzhya region', 'м_Славутич_та_Славутицька_територіальна_громада': 'M_ Slavutich_t_Slavutitskaya_ territorial', 'м_Охтирка_та_Охтирська_територіальна_громада': 'm_htyrka_that_ochtyrskaya_ territorial', 'Тернопільська_область': 'Ternopil region', 'НовоградВолинський_район': 'Novograd Volyn district', 'Бородянська_територіальна_громада': 'Borodyansk Territorial Community', 'Галицинівська_територіальна_громада': 'Halycin territorial community', 'Камяномостівська_територіальна_громада': 'Kamyanostivsky territorial community', 'Новобузька_територіальна_громада': 'Novobuzhka territorial community', 'Нікопольський_район': 'Nikopol district', 'Арбузинська_територіальна_громада': 'Arbuzin territorial community', 'Сквирська_територіальна_громада': 'Skvyr territorial community', 'Одеський_район': 'Odessa_ region', 'Рівненська_область': 'Rivne region', 'Вишгородська_територіальна_громада': 'Vyshgorod territorial community', 'Прибузька_територіальна_громада': 'Pribuzhsky territorial community', 'Волинська_область': 'Volyn region', 'Олександрівська_територіальна_громада': 'Alexander Territorial Community', 'Доманівська_територіальна_громада': 'Semenovskaya territorial community', 'Покровська_територіальна_громада': 'Pokrovskaya territorial community', 'Хмельницька_область': 'Khmelnytsky region', 'м_Нікополь_та_Нікопольська_територіальна_громада': 'm_nikopol_t_nikopol_ territorial_eat', 'Курахівська_територіальна_громада': 'Kurakhiv Territorial Community', 'Добропільська_територіальна_громада': 'Dobropil territorial community', 'м_Рубіжне_та_Рубіжанська_територіальна_громада': 'M_Rebizhne_tay_rubizhan_ territorial', 'м_Фастів_та_Фастівська_територіальна_громада': 'Fastov and Fastiv Territorial Community', 'МішковоПогорілівська_територіальна_громада': 'Mishkovopogorilovskaya_ territorial', 'Веселинівська_територіальна_громада': 'Veselinovsky territorial community', 'Степногірська_територіальна_громада': 'Stepnogorsk territorial community', 'Миколаївський_район': 'Mykolaiv district', 'Новоукраїнський_район': 'Novoukrainskaya', 'м_Чугуїв_та_Чугуївська_територіальна_громада': 'm Chuguev_t_ Chuguev territorial community', 'Богодухівський_район': 'Bohodukhiv district', 'Вільнозапорізька_територіальна_громада': 'Freezporozhian territorial community', 'Краматорський_район': 'Kramatorsk district', 'Волноваська_територіальна_громада': 'Vyshniv Territorial Community', 'м_Кривий_Ріг_та_Криворізька_територіальна_громада': 'm_bravy_g_t_t_kryvyi Rih', 'Сіверська_територіальна_громада': 'Seversky territorial community', 'Синельниківський_район': 'Sinelnikov district', 'Сумський_район': 'Sumy_ district', 'Радомишльська_територіальна_громада': 'Radomyshl Territorial Community', 'м_Запоріжжя_та_Запорізька_територіальна_громада': 'M_ Zaporozhye_tay_zaporozhka_Thertotorial', 'Чернігівська_область': 'Chernihiv region', 'Кременчуцький_район': 'Kremenchug', 'Преображенська_територіальна_громада': 'Transfiguration territorial community', 'Конотопський_район': 'Konotop district', 'Донецький_район': 'Donetsk_ region', 'Жашківська_територіальна_громада': 'Zhashkiv Territorial Community', 'Кіровоградська_область': 'Kirovohrad region', 'м_Миколаїв_та_Миколаївська_територіальна_громада': 'M_mikolaev_ta_mikolaivska_Thertotorial_Com', 'м_Коростень_та_Коростенська_територіальна_громада': 'M_ Corosten_that_corosten_ territorial', 'Шосткинський_район': 'Shostka district', 'м_Бровари_та_Броварська_територіальна_громада': 'm_ brovary_t_brovarskaya_Thertotor', 'Голованівський_район': 'Golovanivskyi district', 'Первомайський_район': 'Pervomaisk district', 'м_Суми_та_Сумська_територіальна_громада': 'm_sum_time_sum_ territorial_ community', 'м_Миргород_та_Миргородська_територіальна_громада': 'm_mirgorod_Thmgorod_ territorial', 'Одеська_область': 'Odesa region', 'Сєвєродонецький_район': 'Severodonetsk district', 'м_Переяслав_та_Переяславська_територіальна_громада': 'M_ Pereyaslav_ta_eretslavskaya_Thertetorial', 'м_Сєвєродонецьк_та_Сєвєродонецька_територіальна_громада': 'M_ Severodonetsk_TA_Severodonetsk_Thertotorial_Com', 'Гайсинський_район': 'Gaysinsky district', 'Попаснянська_територіальна_громада': 'Popasnyansk territorial community', 'Бузька_територіальна_громада': 'The Bug Territorial Community', 'м_Первомайськ_та_Первомайська_територіальна_громада': 'M_Peomaysk_T_PERVOMAYA_THERORTORIORIORIAL COMMUNICATION', 'Марїнська_територіальна_громада': 'Mariinskaya Territorial Community', 'Авдіївська_територіальна_громада': 'Avdiivsky Territorial Community', 'Лозівський_район': 'Lozovsky district', 'м_Словянськ_та_Словянська_територіальна_громада': 'M_ Slovyansk_tay_ Sloviansk_Thertotor', 'Червоногригорівська_територіальна_громада': 'Chervonohrigor Territorial Community', 'Оріхівська_територіальна_громада': 'Orikhiv territorial community', 'Інгульська_територіальна_громада': 'Ingul territorial community', 'Горлівський_район': 'Gorlovsky_ district', 'Березнегуватська_територіальна_громада': 'Berezneguvatsky territorial community', 'Дніпропетровська_область': 'Dnipropetrovsk region', 'Привільненська_територіальна_громада': 'Privolinsky territorial community', 'Узинська_територіальна_громада': 'Uzin territorial community', 'Волноваський_район': 'Volnovaksky District', 'м_Ізюм_та_Ізюмська_територіальна_громада': 'M_IZUM_TA_IZUMA_THERORIORIORIORIAL COMMUNICATION', 'Камянопотоківська_територіальна_громада': 'Kamynotokivskaya_ territorial', 'Вовчанська_територіальна_громада': 'The Vovchansk Territorial Community', 'м_Першотравенськ_та_Першотравенська_територіальна_громада': 'M_ Pershotravensk_A_PPETRORAVENSKA_The community', 'Покровський_район': 'Pokrovsky_ district', 'Житомирська_область': 'Zhytomyr region', 'м_Очаків_та_Очаківська_територіальна_громада': 'm_ochakiv_ta_ochakivska_Thertotor', 'м_Полтава_та_Полтавська_територіальна_громада': 'M_Poltava_ta_Poltavskaya_ territorial', 'Соледарська_територіальна_громада': 'Soledar territorial community', 'м_Маріуполь_та_Маріупольська_територіальна_громада': 'M_Mariupol_t_Mariupolskaya_The community', 'Миколаївська_область': 'Mykolaiv region', 'м_Конотоп_та_Конотопська_територіальна_громада': 'm_ koonotope_t_konotope_ territorial', 'Торецька_територіальна_громада': 'Toretsky territorial community', 'Костянтинівська_територіальна_громада': 'Konstantinovsky territorial community', 'Цебриківська_територіальна_громада': 'Tsebrikovsky territorial community', 'Березівський_район': 'Berezovsky district', 'м_Харків_та_Харківська_територіальна_громада': 'm Kharkiv_ta_harkiv territorial community', 'м_Мелітополь_та_Мелітопольська_територіальна_громада': 'm_melitopol_t_melitopol_ territorial_ community', 'м_Пирятин_та_Пирятинська_територіальна_громада': 'm_piryatin_that_piryatyn_ territorial', 'Кривоозерська_територіальна_громада': 'Kryvozersky territorial community', 'Казанківська_територіальна_громада': 'Zinkiv territorial community', 'Жмеринський_район': 'Zhmeryn district', 'Широківська_територіальна_громада': 'Shirokiv Territorial Community', 'Лиманська_територіальна_громада': 'Liman territorial community', 'БілгородДністровський_район': 'Belgorodnistrovsky_ district', 'Павлоградський_район': 'Pavlograd district'}

df["text_hashtag"] = df["text_hashtag"].replace(translations)

df['date'] = pd.to_datetime(df['date'])

# Create a new dataframe for paired alerts
alerts = []
current_alerts = {}

# Group by location (text_hashtag) and process alerts in chronological order
for location in df['text_hashtag'].unique():
    location_alerts = df[df['text_hashtag'] == location].sort_values('date')
    
    for _, row in location_alerts.iterrows():
        if row['alert_flag'] == 'Start':
            current_alerts[location] = {
                'id': row['id'],
                'text_bold': row['text_bold'],
                'text_hashtag': row['text_hashtag'],
                'alert_start': row['date']
            }
        elif row['alert_flag'] in ['Stop', 'Stop partial'] and location in current_alerts:
            alert_data = current_alerts[location]
            alert_data['alert_end'] = row['date']
            alert_data['duration'] = (row['date'] - alert_data['alert_start']).total_seconds() / 60  # Duration in minutes
            
            # Calculate duration during work hours (7:00-22:00)
            start_time = alert_data['alert_start']
            end_time = alert_data['alert_end']
            
            # Set work hours boundaries for start and end dates
            start_work = start_time.replace(hour=7, minute=0, second=0, microsecond=0)
            end_work = start_time.replace(hour=22, minute=0, second=0, microsecond=0)
            
            # If alert spans multiple days, we need to handle each day
            during_work_minutes = 0
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                # Set work hours for current day
                day_start = pd.Timestamp.combine(current_date, pd.Timestamp("07:00:00").time())
                day_end = pd.Timestamp.combine(current_date, pd.Timestamp("22:00:00").time())
                
                # Calculate overlap for this day
                period_start = max(start_time, day_start)
                period_end = min(end_time, day_end)
                
                # If there is overlap with work hours, add it to total
                if period_start < period_end:
                    overlap = (period_end - period_start).total_seconds() / 60
                    during_work_minutes += overlap
                
                current_date += pd.Timedelta(days=1)
            
            alert_data['during_work_hours'] = during_work_minutes
            alerts.append(alert_data)
            del current_alerts[location]

# Create new dataframe with paired alerts
df_alerts = pd.DataFrame(alerts)

# Sort by start time
df_alerts = df_alerts.sort_values('alert_start')

# Save the new dataframe
df_alerts.to_csv("data/air_alarm_all.csv", index=False, encoding='utf-8-sig')

print("\nAlert statistics:")
print(f"Total number of complete alerts: {len(df_alerts)}")
print("\nSample of the alerts dataframe:")
print(df_alerts.head())