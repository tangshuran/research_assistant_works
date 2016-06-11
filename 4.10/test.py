import numpy as np
path=r"D:\HIWI\python-script\new_new_results\4.10/data10.dat"

phiwinkel=np.array([ 0.        ,  0.01745329,  0.03490659,  0.05235988,  0.06981317,
        0.08726646,  0.10471976,  0.12217305,  0.13962634,  0.15707963,
        0.17453293,  0.19198622,  0.20943951,  0.2268928 ,  0.2443461 ,
        0.26179939,  0.27925268,  0.29670597,  0.31415927,  0.33161256,
        0.34906585,  0.36651914,  0.38397244,  0.40142573,  0.41887902,
        0.43633231,  0.45378561,  0.4712389 ,  0.48869219,  0.50614548,
        0.52359878,  0.54105207,  0.55850536,  0.57595865,  0.59341195,
        0.61086524,  0.62831853,  0.64577182,  0.66322512,  0.68067841,
        0.6981317 ,  0.71558499,  0.73303829,  0.75049158,  0.76794487,
        0.78539816,  0.80285146,  0.82030475,  0.83775804,  0.85521133,
        0.87266463,  0.89011792,  0.90757121,  0.9250245 ,  0.9424778 ,
        0.95993109,  0.97738438,  0.99483767,  1.01229097,  1.02974426,
        1.04719755,  1.06465084,  1.08210414,  1.09955743,  1.11701072,
        1.13446401,  1.15191731,  1.1693706 ,  1.18682389,  1.20427718,
        1.22173048,  1.23918377,  1.25663706,  1.27409035,  1.29154365,
        1.30899694,  1.32645023,  1.34390352,  1.36135682,  1.37881011,
        1.3962634 ,  1.41371669,  1.43116999,  1.44862328,  1.46607657,
        1.48352986,  1.50098316,  1.51843645,  1.53588974,  1.55334303,
        1.57079633,  1.58824962,  1.60570291,  1.6231562 ,  1.6406095 ,
        1.65806279,  1.67551608,  1.69296937,  1.71042267,  1.72787596,
        1.74532925,  1.76278254,  1.78023584,  1.79768913,  1.81514242,
        1.83259571,  1.85004901,  1.8675023 ,  1.88495559,  1.90240888,
        1.91986218,  1.93731547,  1.95476876,  1.97222205,  1.98967535,
        2.00712864,  2.02458193,  2.04203522,  2.05948852,  2.07694181,
        2.0943951 ,  2.11184839,  2.12930169,  2.14675498,  2.16420827,
        2.18166156,  2.19911486,  2.21656815,  2.23402144,  2.25147474,
        2.26892803,  2.28638132,  2.30383461,  2.32128791,  2.3387412 ,
        2.35619449,  2.37364778,  2.39110108,  2.40855437,  2.42600766,
        2.44346095,  2.46091425,  2.47836754,  2.49582083,  2.51327412,
        2.53072742,  2.54818071,  2.565634  ,  2.58308729,  2.60054059,
        2.61799388,  2.63544717,  2.65290046,  2.67035376,  2.68780705,
        2.70526034,  2.72271363,  2.74016693,  2.75762022,  2.77507351,
        2.7925268 ,  2.8099801 ,  2.82743339,  2.84488668,  2.86233997,
        2.87979327,  2.89724656,  2.91469985,  2.93215314,  2.94960644,
        2.96705973,  2.98451302,  3.00196631,  3.01941961,  3.0368729 ,
        3.05432619,  3.07177948,  3.08923278,  3.10668607,  3.12413936,
        3.14159265,  3.15904595,  3.17649924,  3.19395253,  3.21140582,
        3.22885912,  3.24631241,  3.2637657 ,  3.28121899,  3.29867229,
        3.31612558,  3.33357887,  3.35103216,  3.36848546,  3.38593875,
        3.40339204,  3.42084533,  3.43829863,  3.45575192,  3.47320521,
        3.4906585 ,  3.5081118 ,  3.52556509,  3.54301838,  3.56047167,
        3.57792497,  3.59537826,  3.61283155,  3.63028484,  3.64773814,
        3.66519143,  3.68264472,  3.70009801,  3.71755131,  3.7350046 ,
        3.75245789,  3.76991118,  3.78736448,  3.80481777,  3.82227106,
        3.83972435,  3.85717765,  3.87463094,  3.89208423,  3.90953752,
        3.92699082,  3.94444411,  3.9618974 ,  3.97935069,  3.99680399,
        4.01425728,  4.03171057,  4.04916386,  4.06661716,  4.08407045,
        4.10152374,  4.11897703,  4.13643033,  4.15388362,  4.17133691,
        4.1887902 ,  4.2062435 ,  4.22369679,  4.24115008,  4.25860337,
        4.27605667,  4.29350996,  4.31096325,  4.32841654,  4.34586984,
        4.36332313,  4.38077642,  4.39822972,  4.41568301,  4.4331363 ,
        4.45058959,  4.46804289,  4.48549618,  4.50294947,  4.52040276,
        4.53785606,  4.55530935,  4.57276264,  4.59021593,  4.60766923,
        4.62512252,  4.64257581,  4.6600291 ,  4.6774824 ,  4.69493569,
        4.71238898,  4.72984227,  4.74729557,  4.76474886,  4.78220215,
        4.79965544,  4.81710874,  4.83456203,  4.85201532,  4.86946861,
        4.88692191,  4.9043752 ,  4.92182849,  4.93928178,  4.95673508,
        4.97418837,  4.99164166,  5.00909495,  5.02654825,  5.04400154,
        5.06145483,  5.07890812,  5.09636142,  5.11381471,  5.131268  ,
        5.14872129,  5.16617459,  5.18362788,  5.20108117,  5.21853446,
        5.23598776,  5.25344105,  5.27089434,  5.28834763,  5.30580093,
        5.32325422,  5.34070751,  5.3581608 ,  5.3756141 ,  5.39306739,
        5.41052068,  5.42797397,  5.44542727,  5.46288056,  5.48033385,
        5.49778714,  5.51524044,  5.53269373,  5.55014702,  5.56760031,
        5.58505361,  5.6025069 ,  5.61996019,  5.63741348,  5.65486678,
        5.67232007,  5.68977336,  5.70722665,  5.72467995,  5.74213324,
        5.75958653,  5.77703982,  5.79449312,  5.81194641,  5.8293997 ,
        5.84685299,  5.86430629,  5.88175958,  5.89921287,  5.91666616,
        5.93411946,  5.95157275,  5.96902604,  5.98647933,  6.00393263,
        6.02138592,  6.03883921,  6.0562925 ,  6.0737458 ,  6.09119909,
        6.10865238,  6.12610567,  6.14355897,  6.16101226,  6.17846555,
        6.19591884,  6.21337214,  6.23082543,  6.24827872,  6.26573201])
Ephi2_divide_Emax2=np.array([ 0.44897433,  0.42398403,  0.41820004,  0.42637437,  0.43536098,
        0.44014862,  0.43837426,  0.43257252,  0.43879582,  0.45734607,
        0.48815229,  0.51532471,  0.53113308,  0.53166792,  0.51775088,
        0.49910286,  0.50238045,  0.49829912,  0.48659668,  0.46836911,
        0.46561775,  0.4959084 ,  0.54281837,  0.59804421,  0.65142876,
        0.69781523,  0.73588851,  0.75166657,  0.74177088,  0.72382553,
        0.70597685,  0.68563901,  0.65600165,  0.61748252,  0.59994034,
        0.59640577,  0.59598845,  0.59658273,  0.58489634,  0.57024501,
        0.56860267,  0.57079328,  0.5441572 ,  0.50092721,  0.47323831,
        0.44537425,  0.41801101,  0.40625797,  0.4234204 ,  0.46020878,
        0.50621187,  0.54597667,  0.56754922,  0.56288477,  0.53168548,
        0.49204105,  0.46781621,  0.43742915,  0.4031238 ,  0.39721734,
        0.41969948,  0.45164884,  0.49582532,  0.54299529,  0.57756936,
        0.59848572,  0.60883643,  0.60495546,  0.59041686,  0.548673  ,
        0.4894661 ,  0.43935572,  0.41410372,  0.41305209,  0.43379265,
        0.46532423,  0.50430016,  0.53307026,  0.54270608,  0.5552809 ,
        0.56881891,  0.5711235 ,  0.58753144,  0.57714241,  0.54730625,
        0.51779856,  0.49979782,  0.49849623,  0.50873159,  0.51633314,
        0.50552642,  0.47878994,  0.44763539,  0.42800537,  0.42508638,
        0.42116975,  0.41534715,  0.40102492,  0.38482205,  0.37489882,
        0.37537912,  0.38097292,  0.39153172,  0.40626697,  0.42062663,
        0.43545799,  0.45620934,  0.48561422,  0.53171606,  0.57955633,
        0.61357727,  0.6301296 ,  0.63153579,  0.62066786,  0.60312046,
        0.58784444,  0.58882625,  0.5995089 ,  0.62406964,  0.64632952,
        0.65668132,  0.65109067,  0.627128  ,  0.59248364,  0.54940666,
        0.49026247,  0.43070375,  0.39004741,  0.38237702,  0.38690774,
        0.39720996,  0.42048266,  0.47728911,  0.55200917,  0.61521428,
        0.6468219 ,  0.64411157,  0.61287353,  0.56773496,  0.51336916,
        0.45603966,  0.40540434,  0.37577416,  0.40964966,  0.47652866,
        0.54902197,  0.61740142,  0.67512945,  0.71352145,  0.72781336,
        0.71837689,  0.69008402,  0.65213779,  0.61705284,  0.58333915,
        0.55496125,  0.53858118,  0.5407266 ,  0.55963465,  0.5822161 ,
        0.60107494,  0.61151468,  0.61678135,  0.62807639,  0.6392048 ,
        0.64256757,  0.6370261 ,  0.62550959,  0.63558512,  0.66604513,
        0.67583299,  0.65247417,  0.63084089,  0.62868465,  0.6393979 ,
        0.67265522,  0.72229785,  0.74284708,  0.72213127,  0.6695855 ,
        0.60503496,  0.54567183,  0.51304363,  0.51224938,  0.53155193,
        0.55600199,  0.58904246,  0.61928943,  0.66250157,  0.74476314,
        0.82145956,  0.87154442,  0.8861949 ,  0.86792466,  0.8266878 ,
        0.76370217,  0.69842748,  0.64938918,  0.60733201,  0.56020798,
        0.52064451,  0.51211675,  0.5537296 ,  0.61059662,  0.66914149,
        0.72662357,  0.78612181,  0.85389517,  0.92303502,  0.97590412,
        1.        ,  0.99180511,  0.95476482,  0.89984503,  0.83843484,
        0.7660541 ,  0.69514101,  0.63457792,  0.58676018,  0.56544023,
        0.5624669 ,  0.56273842,  0.55975291,  0.55957272,  0.54780117,
        0.52736545,  0.48983001,  0.44443934,  0.40291318,  0.37292335,
        0.37663438,  0.4030123 ,  0.42153421,  0.42919973,  0.4493823 ,
        0.46963367,  0.5045093 ,  0.54829906,  0.57959951,  0.58908458,
        0.58622297,  0.58037049,  0.56979234,  0.5609784 ,  0.54265422,
        0.50555937,  0.45589632,  0.40957033,  0.37400943,  0.34187971,
        0.30817779,  0.30311005,  0.30933653,  0.32578277,  0.37166541,
        0.40303987,  0.4158137 ,  0.42014398,  0.42717454,  0.42548005,
        0.43219204,  0.43367753,  0.42395271,  0.40525669,  0.38942527,
        0.39379459,  0.42824142,  0.46641721,  0.50657888,  0.55739023,
        0.58392279,  0.58647838,  0.57294322,  0.55959975,  0.53930346,
        0.51198334,  0.49332719,  0.49536396,  0.51434689,  0.52154165,
        0.51963821,  0.51316395,  0.49762636,  0.48414903,  0.47343548,
        0.47130046,  0.48061334,  0.49696403,  0.51042237,  0.52967012,
        0.58003791,  0.63812872,  0.69049179,  0.73329396,  0.75729715,
        0.75602378,  0.73406004,  0.69319996,  0.63524625,  0.56889533,
        0.50907471,  0.46094532,  0.43152389,  0.42668185,  0.43521622,
        0.46377036,  0.50596002,  0.55288106,  0.59923853,  0.64927829,
        0.69163291,  0.70899955,  0.69737975,  0.66157173,  0.63462307,
        0.61496502,  0.5907519 ,  0.56719041,  0.54725631,  0.53223386,
        0.5268018 ,  0.5421247 ,  0.58361393,  0.63790088,  0.68876086,
        0.728667  ,  0.75142306,  0.74823669,  0.71769626,  0.67187534,
        0.61870888,  0.55669095,  0.50714351,  0.48111201,  0.47597854,
        0.48145632,  0.50559118,  0.54619932,  0.5961202 ,  0.65265212,
        0.68930337,  0.69849953,  0.67827093,  0.63622073,  0.58487393,
        0.53441331,  0.49367302,  0.48279739,  0.49624901,  0.4985099 ,
        0.48974628,  0.48629146,  0.48857284,  0.50622805,  0.52517998,
        0.54662988,  0.55218251,  0.5404064 ,  0.51355406,  0.47889904])

np.savetxt(path, zip(phiwinkel,Ephi2_divide_Emax2), fmt=['%.4f','%.4f'])