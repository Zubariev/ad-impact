import pandas as pd

df = pd.read_excel('ALL_DATA.xlsx', sheet_name='Sheet1')

df1 = pd.melt(df, id_vars=['Y&W'], value_vars=['Metro_Digital_Display_Sum of Budg_UAH', 
                                              'Metro_Digital_Video_Sum of Budg_UAH', 
                                              'Metro_Digital_Social_Sum of Budg_UAH', 
                                              'Metro_TV_Пряма_Sum of Budg_UAH', 
                                              'Metro Sum of Budg_UAH____', 
                                              'Metro _ OOH_Net_UAH', 
                                              'Metro_Radio Sum of Budg_UAH___'
                                              ], var_name='channel', value_name='budget')
print(df1.head())

df2 = pd.melt(df, id_vars=['Y&W'], value_vars=['Chernihiv__ visit', 
                                               'Chernivtsi__ visit', 
                                               'Dnipro__ visit', 
                                               'Ivano-Frankivsk__ visit', 
                                               'Kharkiv__ visit', 
                                               'Kryvyi Rig__ visit', 
                                               'Kyiv__ visit', 
                                               'Lutsk__ visit', 
                                               'Lviv__ visit', 
                                               'Mykolaiv__ visit', 
                                               'Odesa__ visit', 
                                               'Poltava__ visit', 
                                               'Rivne__ visit', 
                                               'Ternopil__ visit', 
                                               'Vinnytsia__ visit', 
                                               'Zaporizhzhia__ visit', 
                                               'Zhytomyr__ visit'], var_name='city', value_name='visit_count')
print(df2.head())

df3 = pd.merge(df1, df2, on=['Y&W'], how='inner')
print(df3.head())

df3.to_excel('ALL_DATA_LONG.xlsx', index=False)