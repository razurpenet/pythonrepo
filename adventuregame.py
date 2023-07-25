#AUTHOR OLOYEDE RASHEED
#copyright www.razurpe.net
#!/usr/bin/env python
# coding: utf-8

# In[6]:


#GAME NAME: THE ADVENTURE OF PRINCESS LIGHT
def path01():
    print("""There was a village in the year 1932, located in the most accessible part of the
    United Kingdom under the reign of King Mozak II. The king lived with no heir for about
    22years. 
    
    However, in the year 1954 the queen bore a princess who was named Light. Light was never to be 
    left alone in the dark all her life has the monster who casted a spell of barrenness on the queen
    was a monster of the dark.
    
    On Lights's 18th birthday there was merrying all over the kingdom, Light was allowed to go into the 
    villages to have fun and experience the true feeling of being a royal. Light was so happy she went farther
    into the village. Back at the palace there were news flying around about the kingdom been attacked by the 
    monster of the dark and his army
    
    The king quickly summoned the warriors to go into town and get the princess back home in time.
    
    Unfortunately, the attack on the kingdom had begun, the kingdom was already in disarray and every family, man,
    woman and children were scamping for safety. The guards couldnt reach Princess Light, they were struck dead on
    their way
    
    Princess light was nowhere to be found. After a war that lasted 2 days the monsters of the dark retreated. Thankfully, 
    Princess Light was found and returned to the palace
    
    The princess was asleep at around 01:20am. When a loud bang was heard in the palace and surroundings and everywhere went
    dark. The princess got up and the adventure of Princess Light began>>>>>>>>>>>>>>>>>>>>>>>>>""")


# In[7]:


def path1():
    import time
    print("""Princess Light got up the bed and approached the door. She opened the door and she could not see a thing.
    Every where was dark and it was scary. She could hear scary noise far away in the palace and within the kingdom.
    She was scared at the terrifying experience coupled with her experiences outside the palace few days ago when the
    war was raging, she needs to do something.........
    
    1. Step Out -- type st
    2. Call the maid -- type cm
    3. Go back to bed -- type gtb""")
    
    user = input()
    time.sleep(3)
    x = 'correct'
    while(x == 'correct'):
        if(user.upper() == "ST"):
            print("""The princess made the right decision and steps out of her room to find out what is going on.
            To her left is the walkway leading to the kings room while to her right is the walkway to reach out to
            the guards. The princess was confused and did not know where to go..................
            
            1. Go to Kings room --type L to do this
            2. Get to the guards. --type R to do this
            
            """)
            user2 = input()
            if(user2.upper() == "L"):
                time.sleep(2)
                print("""Princess Light opens the door to the kings room, steps in and calls out to her mum and dad!""")
                time.sleep(4)
                path11()
            elif(user2.upper() == "R"):
                time.sleep(4)
                print("""Pricess Light decides to call the guards for safety""")
                time.sleep(4)
                path12()
            x = 'inco'
        elif(user.upper() == "CM"):
            time.sleep(2)
            print("""Princess Light decided to call the maid. However, as a result of the ongoing uproar in the kingdom. 
            The palace maids and guards are no where to be found, not even the king and queen.
            You'll need to step out of the room now or at least do something.........The confusing continues...
            
            
            """)
            time.sleep(5)
            path1()
            x = 'inco'
        elif(user.upper() == "GTB"):
            print("""You have decided to go back to bed and the monster was able to capture you and all the kingdom,
            you have been killed. GAME OVER
            
            Restart game?
            Yes or No?""")
            user1 = input()
            if(user1.upper() == "YES"):
                time.sleep(3)
                path1()
            elif (user1.upper() == "NO"):
                time.sleep(3)
                print("""Thank You for playing ADVENTURE OF PRINCESS LIGHT, See you later. Bye!""")
            else:
                print("""Type Yes or No to proceed""")
                user1 = input()
                x = 'inco'
        else:
            print("""ENTER THE CORRECT CHOICE""")
            user = input()
            
            
        


# In[8]:


def path11():
    import time
    print("""She yelled and yelled continuously in the dark but she wasnt getting any response. The king and the Queen 
    weren't in the room anymore. She became even more confused, scared and shakey.
    
    She could hear faint footsteps approaching from a distance. Quickly, she took cover behind the closet. She sees a light
    glare running around the room. She went numb for a few seconds
    
    Suddenly she was grabbed from her legs and dragged out of the closet. It was the chief guard, he brought
    Princess Light to safety and informed her that the Monster of the dark had captured the King and the Queen.
    
    That it is only someone with the royal blood who can conquer the monster of the dark. This meant that Princess
    Light needed to take up the challenge herself. However, she was not ready and needs to be prepared. She needs to train
    for 3 different stages of attack to be able to conquer the Monster of the dark.....
    
    1. Pick a weapon to fight with.  type W to pick weapon
    2. Go to Training now type T to train
    """)
    
    user3 = input()
    time.sleep(3)
    if (user3.upper() == "W"):
        print("""Choose your weapon below, Be careful what type of weapon you pick. The monster of the dark cannot be
        killed with just an iron. The monster is a very powerful wizard so you'll need a good combination of iron and spell
        portion to overcome the monster. Pick your weapon now
        """)
        time.sleep(2)
        print("""Weapons are loading""")
        time.sleep(4)
        path13()
    elif (user3.upper() == "T"):
        print("""Great, welcome to your training session Kindly pick a preferred weapon to use in warfare against the monster.
        A combination of Spell and well carved iron would be required to defeat the monster and rescue the royals.
        """)
        path13()
    else:
        print("""Please input the right action to progress in your quest""")
        path11()
    
    
    
    


# In[9]:


def path12():
    import time
    print("""Princess light ran to the right side of the corridor, approaching the guards chamber, she fell and draws
    the attention of a creeping monster that was now closing in on her position. 
    
    She became scared and screamed, she opened her eyes and realized she was in the hands of the chief guard who 
    had strucked the creeping creature with his sword
    
    He brought Princess Light to safety and informed her that the Monster of the dark had captured the King and the Queen.
    
    That it is only someone with the royal blood who can conquer the monster of the dark. This meant that Princess
    Light needed to take up the challenge herself. However, she was not ready and needs to be prepared. She needs to train
    for 3 different stages of attack to be able to conquer the Monster of the dark.....
    
    1. Pick a weapon to fight with. ---type W to pick weapon
    2. Go to Training now ----type T to train
    3. Face the monster and rescue the King and Queen ---type FM""")
    
    user5 = input('Type W, T or FM')
    time.sleep(3)
    if user5.upper() == "W":
        print("""Princess needs a weapon to be able to conquer the monster of the dark.
                A combination of Spell and well carved iron would be required to defeat the monster and rescue the royals
                    
                Loading Arsenal..............""")
        time.sleep(5)
        path13()
    elif user5.upper() == "T":
        time.sleep(3)
        print("""Welcome to levelone of your training session""")
        path13()
    else:
        print("""Please input the right action to progress in your quest""")
        path12()
    
    
    
    
    


# In[10]:


#the weapons path
def path13():
    import time
    print("""A combination of Spell and well carved iron would be required to defeat the monster and rescue the royals
    
        1. Axe and a shotgun --type AS to choose
        2. A spear and King Mozak I's sword of fury --type SS to choose
        3. A spell portion and a King Mozak I's Spear --type SPS to choose""")
    user6 = input()
    weapon = "true"
    if user6.upper() == "AS":
        time.sleep(4)
        print("""You have chosen the Axe and the shotgun. You failed the mission as this weapon combination cannot
        defeat the monster of the dark. You have been killed.
        
        HINT: Iron and a well prepared portion will kill the monster
        GAME OVER!!!""")
        path1()
    elif user6.upper() == "SS":
        time.sleep(4)
        print("""You have chosen the Spear and King Mozak's sword. You failed the mission as this weapon combination cannot
        defeat the monster of the dark. You have been killed.
        
        HINT: Iron and a well prepared portion will kill the monster
        GAME OVER!!!""")
        path1()
    elif user6.upper() == "SPS":
        time.sleep(2)
        print("""Great, You have the right weapon to conquer the monster""")
        time.sleep(4)
        print("""Princess Light has found the right weapon to conquer the monster. Now return to training or go and 
        defeat the monster if you are ready
    
        Hint: Train properly in ternary to be able to defeat the monster of the dark 
        
        Loading...........""")
        time.sleep(5)
        path15()
        
    else:
        print("""Input the appropraite option from 1-3 to proceed""")
        path13()
        
    


# In[11]:


def path15():
    import time
    print("""Princess Light has found the right weapon to conquer the monster. Now return to training or go and 
    defeat the monster if you are ready
    
    Hint: Train properly in ternary to be able to defeat the monster of the dark """)
    time.sleep(5)
    path125()


# In[12]:


def path125():
    import time
    print("""She needs to train for 3 different stages of attack to be able to conquer the Monster of the dark.....
    
    2. Go to Training now ----type T to train
    3. Face the monster and rescue the King and Queen ---type FM""")
    
    user5 = input('Type T or FM')
    time.sleep(3)
    if user5.upper() == "T":
        print("""Welcome to levelone of your training session..............\n""")
        time.sleep(5)
        path16()
    elif user5.upper() == "FM":
        time.sleep(3)
        print("""You decided to face the monster of the dark. You havent had enough training to over the monster.
        You have been killed.
        
        GAME OVER!
        Restart game
        Yes or No?""")
        user1 = input()
        if(user1.upper() == "YES"):
            time.sleep(3)
            path1()
            
        elif (user1.upper() == "NO"):
            time.sleep(3)
            print("""Thank You for playing ADVENTURE OF PRINCESS LIGHT, See you later. Bye!""")
        else:
            print("""Type Yes or No to proceed""")
            user1 = input()
    else:
        print("""...................INPUT T or FM to proceed............................\n""")
        time.sleep(3)
        path125()


# In[13]:


def path16():
    import time
    print("""Would you like to acquire the weapon use skill for defeating a callous monster?""")
    #userme = ["yes", "y", "YES", "Yes"]
    
    user7 = input()
    
    if user7.upper() == "YES":
        print("""Congratulations, You have acquired your first combat skill! Go further in your quest""")
        path17()
    elif user7.upper() == "NO":
        print("""You need to complete your training, but you choose not to. GAME OVER!""")
        path1()
    else:
        print("""...............Please input the correct response..................\n""")
        path16()


# In[14]:


def path17():
    import time
    print("""You have successfully acquire the first combat skill. Would you like to acquire the spell
    kill skill also?""")
    user8 = input()
    if user8.upper() == "YES":
        time.sleep(3)
        print("""Congratulations, you have acquired skill two in your combat rank. The spell skill. Proceed
        now in your quest to conquer the monster of the dark""")
        
        print("""\nLoading.......""")
        path18()
    elif user8.upper() == "NO":
        print("""You need to complete your training, but you choose not to. GAME OVER!""")
        path1()
    else:
        print("""...............Please input the correct response..................\n""")
        path17()
    


# In[15]:


def path18():
    import time
    print("""You have successfully acquird the spell kill. Would you like to acquire the last
    kill skill?""")
    
    user8 = input()
    if user8.upper() == "YES":
        time.sleep(3)
        print("""Congratulations, you have acquired the final skill, The combo kill. Proceed
        now in your quest to conquer the monster of the dark""")
        
        print("""\nLoading.......""")
        path19()
    elif user8.upper() == "NO":
        print("""You need to complete your training, but you choose not to. GAME OVER!""")
        path1()
    else:
        print("""...............Please input the correct response..................\n""")
        path18()
    


# In[16]:


def path19():
    import time
    print("""She needs to train for 3 different stages of attack to be able to conquer the Monster of the dark.....
    
    2. Go to Training now ----****COMPLETED****
    3. Face the monster and rescue the King and Queen ---type FM""")
    
    user5 = input('Type T or FM')
    time.sleep(3)
    if user5.upper() == "T":
        print("""Your training session and preparation has been completed you can now proceed to save your parents
        
        BEST OF LUCK WARRIOR!..............\n""")
        time.sleep(5)
        path19()
    elif user5.upper() == "FM":
        time.sleep(3)
        print("""
        
        You decided to face the monster of the dark. Do you want to proceed?
        Yes or No?
        """)
        user = input()
        if user.upper() == "YES":
            time.sleep(2)
            path21()
        elif user.upper() == "NO":
            print("...........You are still not ready,  well thats understandable..........")
            time.sleep(2)
            path19()
        else:
            print("""
            
            ...................INPUT Yes or No to proceed............................\n""")
            path19()
        
    else:
        print("""...................INPUT T or FM to proceed............................\n""")
        time.sleep(3)
        path19()


# In[17]:


def path21():
    import time
    print("""Welcome to the battle level. Your battle with the monster of the dark is about to begin
    
    Get ready.......""")
    time.sleep(7)
    print("""Princess Light, armed with her weapons steps out in the dark and calls out the monster of the dark
    to let go of the king and the queen. She yelled in anger and waited for the monster""")
    time.sleep(14)
    print("""Monster: You princess will dare not fight me. I am your nightmare and you will be killed by my sword
    
    **The monster jumps out of the dark and tries to attack the princess. 
    
    Princess Light fought hard and long with the monster, she was so brave and courageous, she knew she had to do all
    in her power and might to conquer this monster and rescue are parents
    
    The monster became furious after being unable to kill the princess and transformed into a demonic creature with
    7 eyes bearing a single horn in the center of the head and spitting fire
    
    The princess became scared and jump few steps back after almost being consumed by the fire. She didnt know what to
    do anymore, she immediately remembered what she learnt during her training
    
    She decided to use her concealed weapons 
    
    Which weapon combination will defeat the monster immediately?
    
    1. A sword and a combo kill skill --type SC
    2. A spear and King Mozak I's sword of fury --type SS to choose
    3. A spell portion and a King Mozak I's Spear --type SPS to choose
    """)
    time.sleep(20)
    user = input()
    if user.upper() == "SPS":
        time.sleep(1)
        print("""Princess light brought out the Ancient spear of King Mozak, her grandfather and the spell, she
              manuevered her way skillfully defending herself and blaring double kill combo attack she was
              trained in""")
        time.sleep(3)
        print("""She dipped the spear head into the spellportion thereby making the spear poisonous
        she landed a heavy strike on the monsters chest,the spear went through the monsters heart and the 
        monster became green and shinning, the monster gave up. 
        
        Princess quickly rushed inthe direction which the monster kept theking and the queen in captivity
        she could see her father, she rescued him and brought him to safety. 
        
        She could see her mum! she turned and found her mum in the die and weak arms of the monster""")
        time.sleep(10)
        path20()
    elif user.upper() == "SC":
        time.sleep(2)
        print("""Princess light picked the wrong weapon. GAME OVER
        
        Restart game?
        Yes or No?""")
        
        user1 = input()
        if(user1.upper() == "YES"):
            time.sleep(3)
            path1()
            
        elif (user1.upper() == "NO"):
            time.sleep(3)
            print("""Thank You for playing ADVENTURE OF PRINCESS LIGHT, See you later. Bye!""")
        else:
            print("""Type Yes or No to proceed""")
            user1 = input()
    elif user.upper() == "SS":
        time.sleep(2)
        print("""Princess light picked the wrong weapon. GAME OVER
        
        Restart game?
        Yes or No?""")
        
        user1 = input()
        if(user1.upper() == "YES"):
            time.sleep(3)
            path1()
            
        elif (user1.upper() == "NO"):
            time.sleep(3)
            print("""Thank You for playing ADVENTURE OF PRINCESS LIGHT, See you later. Bye!""")
        else:
            print("""Type Yes or No to proceed""")
            user1 = input()       
    else:
        print("""Please input the right action to proceed""")
        user = input()
        
    


# In[18]:


def path20():
    import time
    user = input("Type pick to pick up your mum")
    if user.upper() == "PICK":
        time.sleep(3)
        print("""Princess light, running towards her mum and screamed mother! mother!! mother!!!
        As she was getting nearer a bright red light overwhelmed her and the monster was able to create a
        portal. The monster varnished with the queen, princess criedand screamed but she couldn't do anything
        anymore
        
        
        
        ..........................END OF CHARPTER 1............................""")
    elif user.upper() == "Run":
        time.sleep(3)
        print("""You were able to save the king but unfortunately, you couldnt save your mother 
        
        You failed the mission, GAME OVER!!!""")
    else:
        print("""Please input the appropraite action""")
        path20()


# In[ ]:


path01()
path1()


# In[ ]:




