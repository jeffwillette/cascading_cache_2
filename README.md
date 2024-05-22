# Cascading KV Cache

## How to Install

```bash
conda env create -f environment.yml
pip install -e .

# test that triton can compile cache correctly
python cascade/models/sink_cache_cascade.py
```

## RUN PG19

```
# set parameters for desired experiment in ./test-pg19.bash
./text-pg19.bash
```

## Run LongBench

```
cd third_party/LongBench-timber/

# edit run.sh to run the correct models/datasets
./run.sh
```

## Run MMLU

```
# set parameters for desired experiment in ./test-mmlu.bash
./test-mmlu.bash
```


## Note

PG19 BOOK TITLES

```
0 Reminiscences of Pioneer Days in St. Paul by Frank Moore
1 Dragon's blood by Henry Milner Rideout
2 Travels in Morocco Volume 2 by James Richardson
3 Impressions of Theophrastus Such by George Eliot
4 Odd Craft Part 4: Bill's Lapse by W.W. Jacobs
5 The S. W. F. Club by Caroline E. Jacobs
6 Frank Merriwell Down South by Burt L. Standish
7 Critical Miscellanies (Vol. 2 of 3) by John Morley
8 From Sand Hill to Pine by Bret Harte
9 Child's Health Primer For Primary Classes by Jane Andrews
10 On the Portraits of English Authors on Gardening by Samuel Felton
11 Discourse of a Method for the Well Guiding of Reason by Rene Descartes
12 Laurence Sterne in Germany by Harvey Waterman Thayer
13 The Forester's Daughter by Hamlin Garland
14 The Life of Gordon Volume II by Demetrius Charles Boulger
15 Minnie's Pet Monkey by Madeline Leslie
16 In Her Own Right by John Reed Scott
17 Turn About Eleanor by Ethel M. Kelley
18 Jennie Gerhardt by Theodore Dreiser
19 Red White Blue Socks Part Second by Sarah L. Barrow
20 The 'Patriotes' of '37 by Alfred D. Decelles
21 Carmen Ariza by Charles Francis Stocking
22 Vestiges of the Mayas by Augustus Le Plongeon
23 The Stones of Venice Volume I (of 3) by John Ruskin
24 The Church: Her Books and Her Sacraments by E. E. Holmes
25 The Real Latin Quarter by F. Berkeley Smith
26 Try Again by Oliver Optic
27 The Story of Pocahantas by Charles Dudley Warner
28 Last Days of the Rebellion by Alanson M. Randol
29 The Patrol of the Sun Dance Trail by Ralph Connor
30 Caxton's Book: A Collection of Essays Poems Tales and
31 The Wandering Jew Book 2 by Eugene Sue
32 Reminiscences of the Great Mutiny 1857-59 by William Forbes-Mitchell
33 Village Life in America 1852-1872 by Caroline Cowles Richards
34 In Camp With A Tin Soldier by John Kendrick Bangs
35 Who? by Elizabeth Kent
36 Arne; A Sketch of Norwegian Country Life by Bj?tjerne Bj?on
37 The Ragged Trousered Philanthropists by Robert Tressell
38 The Inflexible Captive by Hannah More
39 Fire Cloud by Samuel Fletcher
40 Medical Life in the Navy by Gordon Stables
41 Notes and Queries Number 85 June 14 1851 by Various
42 The Fascinating Boston by Alfonso Josephs Sheafe
43 The Wonders Of Instinct by J. H. Fabre
44 The Life Of Thomas Paine Vol. II. (of II) by Moncure Daniel Conway
45 Ingersollia by Robert G. Ingersoll
46 Memoirs of Marie Antoinette Queen Of France Vol. 7
47 Quacks and Grafters by Unknown
48 Prairie Farmer Vol. 56: No. 12 March 22 1884 by Various
49 The Leavenworth Case by Anna Katharine Green
50 Memoir of Hendrick Zwaardecroon commandeur of Jaffnapatam
51 Dandy Dick by Arthur Pinero
52 The Diary of Samuel Pepys June July & August 1661
53 Toto's Merry Winter by Laura E. Richards
54 Oom Paul's People by Howard C. Hillegas
55 The Diary of John Evelyn Volume II (of 2) by John Evelyn
56 The Civil War Through the Camera by Henry William Elson
57 None so Deaf as Those Who Won't Hear by Herbert Pelham Curtis
58 Punch or the London Charivari Vol.107 September 1 1894 by Various
59 How to Solve Conundrums by Anonymous
60 An Unsentimental Journey through Cornwall by Dinah Maria Craik
61 A System of Practical Medicine By American Authors Vol. II by Various
62 Secret Inheritance Volume 2 of 3 by Benjamin Leopold Farjeon
63 Scott's Lady of the Lake by Walter Scott
64 A Treatise on Painting by Leonardo Da Vinci
65 The Growth of the English Constitution by Edward A. Freeman
66 Pen Pictures by B. F. Craig
67 Birds and all Nature Vol. IV No. 3 September 1898 by Various
68 The Amores or Amours by Ovid
69 Notes on Railroad Accidents by Charles Francis Adams
70 The Shire Horse in Peace and War by J. Albert Frost
71 General Nelson's Scout by Byron A. Dunn
72 The Sorrows of Belgium by Leonid Andreyev
73 The Flying Machine Boys in the Wilds by Frank Walton
74 Dr. Elsie Inglis by Lady Frances Balfour
75 Dan The Newsboy by Horatio Alger Jr.
76 The Crisis Complete by Winston Churchill
77 Musical Instruments by Carl Engel
78 The Irish Penny Journal Vol. 1 No. 28 January 9 1841 by Various
79 The Lawton Girl by Harold Frederic
80 A History of Architecture in all Countries Volume 1 3rd ed.
81 Ruby by Molly E. Jamieson
82 Life in the Grey Nunnery at Montreal by Sarah J Richardson
83 The Rover Boys in the Jungle by Arthur M. Winfield
84 The Details of the Rocket System by William Congreve
85 The Good Hope by Herman Heijermans Jr.
86 An Astronomer's Wife by Angelo Hall
87 Another Brownie Book by Palmer Cox
88 Gallegher and Other Stories by Richard Harding Davis
89 Oh Money! Money! by Eleanor Hodgman Porter
90 Nature's Serial Story by E. P. Roe
91 Old Mortality Complete by Sir Walter Scott
92 Coningsby by Benjamin Disraeli
93 The Fair Maid of Perth by Sir Walter Scott
94 India's Love Lyrics by Laurence Hope
95 Scientific American Supplement No. 360 November 25 1882 by Various
96 Baby Mine by Margaret Mayo
97 The Vision of Hell Part 10 by Dante Alighieri
98 A Doctor of the Old School Part 1 by Ian Maclaren
99 K by Mary Roberts Rinehart
```

MMLU
 
 ```
> The following are multiple choice questions (with answers) about business_ethics.                                                                                                                        
                                                                                                                                                                                                           
1. Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.        
                                                  
A. Externalities, Power, Independence                                                                                                                                                                      
B. Publicity, Insubstantial resources, Mutual dependence
C. Publicity, Power, Independence                                                                                                                                                                          
D. Externalities, Power, Mutual dependence
                                                                                                                                                                                                           
Answer:D                                          
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.
                                                                                                     
2. _______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.
                                                                                                     
A. Corporate social responsibility                
B. Business ethics management
C. Sustainability                                                                                    
D. Environmental management           
                                                                                                     
Answer:B                                                                                             
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.                                                                                                                        
                                                                                                     
3. To ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _
________ time period as well as being appointed _________.
                                                                                                     
A. Outside, Limited, Independently        
B. Inside, Limited, Intermittently                                                                   
C. Outside, Unlimited, Intermittently
D. Inside, Unlimited, Independently                                                                  
                                                                                                     
Answer:A                                                                                             
                                                                                                                                                                                                           
> The following are multiple choice questions (with answers) about business_ethics.                  
                                                                                                     
4. Three contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's ope
rations or ________, often involving some form of _______.                                           
                                                  
A. Non-violent direct action, Violent direct action, Indirect action, Boycott                        
B. Indirect action, Instrumental action, Non-violent direct action, Information campaign
C. Indirect action, Violent direct action, Non-violent direct-action Boycott                         
D. Non-violent direct action, Instrumental action, Indirect action, Information campaign
                                                                                                                                                                                                           
Answer:C                                                                                                                                                                                                   
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.                  
                                                  
5. In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate 
the company in achieving _________ . 
                                                                                                     
A. Buycotts, Boycotts, Blockchain technology, Charitable donations
B. Buycotts, Boycotts, Digital technology, Increased Sales                                                                                                                                                 
C. Boycotts, Buyalls, Blockchain technology, Charitable donations
D. Boycotts, Buycotts, Digital technology, Increased Sales                         
                                                  
Answer:D                                                                                                                                                                                                   
                                                                                                     
> I want you to answer the following are multiple choice question about business_ethics.
                                                                                                     
6. _______ such as bitcoin are becoming increasingly mainstream and have a whole host of associated ethical implications, for example, they are______ and more ______. However, they have also been used to
 engage in _______.                                                                                  
                                                                                                     
A. Cryptocurrencies, Expensive, Secure, Financial Crime                                              
B. Traditional currency, Cheap, Unsecure, Charitable giving
C. Cryptocurrencies, Cheap, Secure, Financial crime                                                                                                                                                        
D. Traditional currency, Expensive, Unsecure, Charitable giving                    
                                                  
Answer:
 ```
