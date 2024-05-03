# TimberAttention

## How to Install

```bash
conda env create -f environment.yml
pip install -e .

# test that triton can compile cache correctly
python timber/models/sink_cache_cascade.py
```

## Run LongBench

```
cd third_party/LongBench-timber/

# edit run.sh to run the correct models/datasets
./run.sh
```

## TODO


## Note

ORIGINAL MMLU
 
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
 
 
REVERSED MMLU


```
I want you to answer a question, but first I will show you some examples of similar questions. The question I am interested in is:                                                                         
                                                                                                     
> I want you to answer the following are multiple choice question about business_ethics.                                                                                                                   
                                                  
1. _______ such as bitcoin are becoming increasingly mainstream and have a whole host of associated ethical implications, for example, they are______ and more ______. However, they have also been used to
 engage in _______.                               
                                                                                                     
A. Cryptocurrencies, Expensive, Secure, Financial Crime                            
B. Traditional currency, Cheap, Unsecure, Charitable giving                                          
C. Cryptocurrencies, Cheap, Secure, Financial crime                                                                                                                                                        
D. Traditional currency, Expensive, Unsecure, Charitable giving                                      
                                                  
Answer:<|YOUR ANSWER|>       
                                                                                                     
The examples of similar questions are:
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.                  
                                                                                                     
2. Beyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.        
                                                                                                     
A. Externalities, Power, Independence                                                                                                                                                                      
B. Publicity, Insubstantial resources, Mutual dependence  
C. Publicity, Power, Independence                                                                    
D. Externalities, Power, Mutual dependence
                                                                                                     
Answer:D                             
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.                  
                                                                                                     
3. _______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.                                                         
                                                                                                     
A. Corporate social responsibility                                                                   
B. Business ethics management                                                                                                                                                                              
C. Sustainability                                                                                    
D. Environmental management                       
                                                                                                     
Answer:B                                                                                             
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.     
                                                                                                                                                                                                           
4. To ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _
________ time period as well as being appointed _________.                                           
                                                                                                     
A. Outside, Limited, Independently                
B. Inside, Limited, Intermittently                                                                                                                                                                         
C. Outside, Unlimited, Intermittently
D. Inside, Unlimited, Independently                                                                  
                                                                                                     
Answer:A                                                                                                                                                                                                   
                                                                                                     
> The following are multiple choice questions (with answers) about business_ethics.
                                                  
5. Three contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's ope
rations or ________, often involving some form of _______.                                           
                                                                                                     
A. Non-violent direct action, Violent direct action, Indirect action, Boycott                        
B. Indirect action, Instrumental action, Non-violent direct action, Information campaign                                                                                                                   
C. Indirect action, Violent direct action, Non-violent direct-action Boycott                         
D. Non-violent direct action, Instrumental action, Indirect action, Information campaign             
                                                                                                     
Answer:C                                                                                             
                                                                                                                                                                                                           
> The following are multiple choice questions (with answers) about business_ethics.
                                                  
6. In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate 
the company in achieving _________ .              
                                                  
A. Buycotts, Boycotts, Blockchain technology, Charitable donations                                   
B. Buycotts, Boycotts, Digital technology, Increased Sales                                                                                                                                                 
C. Boycotts, Buyalls, Blockchain technology, Charitable donations                                                                                                                                          
D. Boycotts, Buycotts, Digital technology, Increased Sales                         
                                                                                                     
Answer:D                                                                                                                                                                                                   
                                                                                                                                                                                                           
I want to know the answer to question 1. The answer (<|YOUR ANSWER|>) to the question I am interested in is:
```
