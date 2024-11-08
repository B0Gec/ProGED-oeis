def a079034(n):
    return (n ** 4 - n ** 2 + 12) / 12

sepa = 140
print(str(list(map(a079034, [i for i in range(120)])))[:sepa])
print(str(list(map(a079034, [i for i in range(120)])))[sepa:2*sepa])
print(str(list(map(a079034, [i for i in range(120)])))[2*sepa:])  

non_manuals = ['23167_A169198.txt', '23917_A170320.txt', '03322_A016835.txt', '24141_A170544.txt', '24240_A170643.txt', '24001_A170404.txt', '24014_A170417.txt', '23207_A169238.txt', '22912_A168943.txt', '03330_A016844.txt', '23872_A170275.txt', '22983_A169014.txt', '24006_A170409.txt', '24211_A170614.txt', '15737_A105944.txt', '24053_A170456.txt', '23488_A169519.txt', '23306_A169337.txt', '22856_A168887.txt', '23049_A169080.txt', '23980_A170383.txt', '23742_A170145.txt', '23109_A169140.txt', '06659_A035798.txt', '23860_A170263.txt', '23800_A170203.txt', '23649_A170052.txt', '23219_A169250.txt', '23682_A170085.txt', '06706_A035871.txt', '23720_A170123.txt', '31181_A279282.txt', '23382_A169413.txt', '24034_A170437.txt', '24192_A170595.txt']

sindy_non_manuals = ['05101_A024347.txt', '25707_A188270.txt', '29660_A257293.txt', '32300_A294799.txt', '04581_A021502.txt', '18476_A134465.txt', '27737_A219531.txt', '19050_A140674.txt', '02145_A014084.txt', '17002_A119284.txt', '01790_A011851.txt', '19582_A151626.txt', '30408_A267207.txt', '11353_A058331.txt', '01812_A011879.txt', '12684_A077415.txt', '06545_A034265.txt', '09627_A042791.txt', '32681_A302710.txt', '19335_A144930.txt', '00831_A006096.txt', '22715_A168746.txt', '06210_A029378.txt', '06350_A031878.txt', '19866_A154223.txt', '09608_A042772.txt', '28975_A246640.txt', '03322_A016835.txt', '00670_A005023.txt', '04936_A023470.txt', '22035_A166443.txt', '32291_A294767.txt', '20072_A155648.txt', '28785_A242971.txt', '08631_A041653.txt']


import pandas as pd
csv_filename = 'linear_database_newbl.csv'
# csv = pd.read_csv(csv_filename, low_memory=False, nrows=0)
csv = pd.read_csv(csv_filename, low_memory=False, usecols=['A189743'])
print(csv)


