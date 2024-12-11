import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from collections import Counter

from selenium.webdriver.support.wait import WebDriverWait

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('capabilities={"acceptInsecureCerts": True}')
#TODO CHECK IF NEEDED
chrome_options.add_argument("--disable-images")
chrome_options.add_argument("--disable-javascript")

MISSING_GAMES = [['ESPN', 'IND @ MIL'], ['TNT', 'NOP @ LAL']]


def main():
    page_url_1_6 = 'https://web.archive.org/web/20230824103411/https://www.nba.com/schedule?cal=all&region=1&season=Regular%20Season'
    page_url_7_14 = 'https://web.archive.org/web/20231203164216/https://www.nba.com/schedule'
    page_url_15_26 = 'https://web.archive.org/web/20240128072946/https://www.nba.com/schedule'
    driver = webdriver.Chrome(options=chrome_options)
    print("Loading page 1/3")
    try:
        driver.get(page_url_1_6)
    except:
        driver.quit()
        print("driver error 1-6")
    WebDriverWait(driver, 10).until(
        lambda driver: driver.execute_script("return document.readyState") == "complete"
    )
    logos_extracted_total = []
    desc_extracted_total = []
    logos_extracted_1 = []
    desc_extracted_1 = []
    logos_extracted_2 = []
    desc_extracted_2 = []
    logos_extracted_3 = []
    desc_extracted_3 = []
    print("Page fully loaded!")
    week_num = 1
    with open('../full season data/rs_nbadotcom_links.txt', mode='r', newline='') as file:
        while week_num <= 6:
            dropdown_button = driver.find_element(By.XPATH, '//select[@name="cal"]')
            #print("ok1")
            dropdown_button.click()
            #print("ok2")
            option = driver.find_element(By.XPATH, '//option[@value="'+str(week_num)+'"]')
            #print("ok3")
            option.click()
            #print("ok")
            day_elements = driver.find_elements(By.CLASS_NAME, "ScheduleDay_sd__GFE_w")
            for day in day_elements:
                parent_elements = day.find_elements(By.CLASS_NAME, "ScheduleGame_sg__RmD9I")
                for p_element in parent_elements:
                    try:
                        img_element = p_element.find_element(By.TAG_NAME, 'img')
                    except:
                        print("image not found\n")
                        continue
                    bc = img_element.get_attribute('title')
                    try:
                        team_element = p_element.find_element(By.CLASS_NAME, 'TabLink_tab__uKOPj').find_element(By.TAG_NAME,
                                                                                                                'a')
                    except:
                        print("team element not found\n")
                    try:
                        content = team_element.get_attribute('data-content')
                    except:
                        print("error extracting team\n")
                    assert content != "TBD @ TBD", "EEEEROR"
                    logos_extracted_1.append(bc)
                    desc_extracted_1.append(content)
            week_num += 1
        logos_extracted_total += logos_extracted_1
        desc_extracted_total += desc_extracted_1
        print("games done so far: " + str(len(logos_extracted_total)))
        for desc, link in zip(desc_extracted_1, file):
            assert desc.lower().split()[0] in link and desc.lower().split()[2] in link, "link: " + link + ' content: ' + desc + "all content: " + ' '.join(desc_extracted_1)
        print("1/3 order verified")
        print("\nfinished weeks 1-6\n")
        print("Loading page 2/3")
        try:
            driver.get(page_url_7_14)
        except:
            driver.quit()
            print("driver error 7-14")
        WebDriverWait(driver, 10).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        print("Page fully loaded!")
        dropdown_button = driver.find_element(By.XPATH, '//select[@name="season"]')
        # print("ok1")
        dropdown_button.click()
        # print("ok2")
        option = driver.find_element(By.XPATH, '//option[@value="Regular Season"]')
        # print("ok3")
        option.click()
        # print("ok")
        while week_num <= 14:
            dropdown_button = driver.find_element(By.XPATH, '//select[@name="cal"]')
            #print("ok1")
            dropdown_button.click()
            #print("ok2")
            option = driver.find_element(By.XPATH, '//option[@value="'+str(week_num)+'"]')
            #print("ok3")
            option.click()
            #print("ok")
            try:
                day_elements = driver.find_elements(By.CLASS_NAME, "ScheduleDay_sd__GFE_w")
            except:
                print("days not found")
            for day in day_elements:
                day_date_element = day.find_element(By.CLASS_NAME, 'ScheduleDay_sdDay__3s2Xt')
                day_date = day_date_element.text.lower()
                if day_date == 'thursday, december 7':
                    logos_extracted_2.append(MISSING_GAMES[0][0])
                    desc_extracted_2.append(MISSING_GAMES[0][1])
                    logos_extracted_2.append(MISSING_GAMES[1][0])
                    desc_extracted_2.append(MISSING_GAMES[1][1])
                    continue

                parent_elements = day.find_elements(By.CLASS_NAME, "ScheduleGame_sg__RmD9I")
                for p_element in parent_elements:
                    try:
                        img_element = p_element.find_element(By.TAG_NAME, 'img')
                    except:
                        print("image not found\n")
                        continue
                    bc = img_element.get_attribute('title')
                    try:
                        team_element = p_element.find_element(By.CLASS_NAME, 'TabLink_tab__uKOPj').find_element(By.TAG_NAME,'a')
                    except:
                        print("team element not found\n")
                    try:
                        content = team_element.get_attribute('data-content')
                    except:
                        print("error extracting team\n")
                    assert content != "TBD @ TBD" or day_date == "friday, december 8", day_date
                    if day_date == "friday, january 12":
                        if content == "CHA @ SAS":
                            continue
                        if content == "TOR @ UTA":
                            continue
                    if day_date == 'wednesday, january 17':
                        if content == 'BKN @ POR':
                            continue
                        if content == 'GSW @ UTA':
                            continue
                    if day_date == 'friday, january 19':
                        if content == 'DAL @ GSW':
                            continue
                    if day_date == 'wednesday, january 24':
                        if content == 'PHX @ DAL':
                            continue
                    if day_date == 'sunday, january 28':
                        if content == 'OKC @ DET':
                            continue

                    logos_extracted_2.append(bc)
                    desc_extracted_2.append(content)
                    if content == 'WAS @ BKN' and day_date == "friday, december 8":
                        logos_extracted_2.append('LEAGUE PASS')
                        desc_extracted_2.append('NYK @ BOS')
                    if content == 'HOU @ DEN' and day_date == "friday, december 8":
                        logos_extracted_2.append('LEAGUE PASS')
                        desc_extracted_2.append('SAC @ PHX')
                    if day_date == "friday, january 12":
                        if content == "POR @ MIN":
                            logos_extracted_2.append('ESPN')
                            desc_extracted_2.append('CHA @ SAS')
                            logos_extracted_2.append('LEAGUE PASS')
                            desc_extracted_2.append('TOR @ UTA')
                    if day_date == 'wednesday, january 17':
                        if content == 'DAL @ LAL':
                            logos_extracted_2.append('LEAGUE PASS')
                            desc_extracted_2.append('BKN @ POR')
                    if day_date == 'wednesday, january 24':
                        if content == 'CLE @ MIL':
                            logos_extracted_2.append('ESPN')
                            desc_extracted_2.append('PHX @ DAL')
                    if day_date == 'saturday, january 27':
                        if content == 'SAC @ DAL':
                            logos_extracted_2.append('LEAGUE PASS')
                            desc_extracted_2.append('OKC @ DET')

            week_num +=1
        logos_extracted_total += logos_extracted_2
        desc_extracted_total += desc_extracted_2
        print("games done so far: " + str(len(logos_extracted_total)))
        for desc, link in zip(desc_extracted_2, file):
            assert desc.lower().split()[0] in link and desc.lower().split()[2] in link, "link: " + link + ' content: ' + desc + "all content: " + ' '.join(desc_extracted_2)
        print("2/3 order verified")
        print("\nfinished weeks 7-14\n")
        print("Loading page 3/3")
        try:
            driver.get(page_url_15_26)
        except:
            driver.quit()
            print("driver error 15-26")
        WebDriverWait(driver, 10).until(
            lambda driver: driver.execute_script("return document.readyState") == "complete"
        )
        print("Page fully loaded!")
        dropdown_button = driver.find_element(By.XPATH, '//select[@name="season"]')
        # print("ok1")
        dropdown_button.click()
        # print("ok2")
        option = driver.find_element(By.XPATH, '//option[@value="Regular Season"]')
        # print("ok3")
        option.click()
        # print("ok")
        while week_num <= 26:
            dropdown_button = driver.find_element(By.XPATH, '//select[@name="cal"]')
            #print("ok1")
            dropdown_button.click()
            #print("ok2")
            option = driver.find_element(By.XPATH, '//option[@value="'+str(week_num)+'"]')
            #print("ok3")
            option.click()
            #print("ok")
            try:
                day_elements = driver.find_elements(By.CLASS_NAME, "ScheduleDay_sd__GFE_w")
            except:
                print("days not found")
            for day in day_elements:
                try:
                    day_date_element = day.find_element(By.CLASS_NAME, 'ScheduleDay_sdDay__3s2Xt')
                    day_date = day_date_element.text.lower()
                except:
                    print("day date not found")
                parent_elements = day.find_elements(By.CLASS_NAME, "ScheduleGame_sg__RmD9I")
                for p_element in parent_elements:
                    try:
                        img_element = p_element.find_element(By.TAG_NAME, 'img')
                    except:
                        print("image not found\n")
                        continue
                    bc = img_element.get_attribute('title')
                    try:
                        team_element = p_element.find_element(By.CLASS_NAME, 'TabLink_tab__uKOPj').find_element(By.TAG_NAME,'a')
                    except:
                        print("team element not found\n")
                    try:
                        content = team_element.get_attribute('data-content')
                    except:
                        print("error extracting team\n")
                    assert content != "TBD @ TBD", day_date
                    if day_date == "friday, february 23":
                        if content == 'WAS @ OKC' or content == "MIA @ NOP":
                            continue
                    if day_date == 'tuesday, april 2':
                        if content == 'GSW @ DAL':
                            continue
                    logos_extracted_3.append(bc)
                    desc_extracted_3.append(content)
                    if day_date == "thursday, february 15":
                        if content == "MIL @ MEM":
                            logos_extracted_3.append('LEAGUE PASS')
                            desc_extracted_3.append('GSW @ UTA')
                    if day_date == "friday, february 23":
                        if content == "LAC @ MEM":
                            logos_extracted_3.append('LEAGUE PASS')
                            desc_extracted_3.append('WAS @ OKC')
                            logos_extracted_3.append('ESPN')
                            desc_extracted_3.append('MIA @ NOP')
                    if day_date == "saturday, february 24":
                        logos_extracted_3.append('ABC')
                        desc_extracted_3.append('BOS @ NYK')
                        logos_extracted_3.append('LEAGUE PASS')
                        desc_extracted_3.append('BKN @ MIN')
                        break
                    if day_date == 'tuesday, april 2':
                        if content == 'LAC @ SAC':
                            logos_extracted_3.append('TNT')
                            desc_extracted_3.append('DAL @ GSW')
                    if day_date == "friday, april 5":
                        if content == 'SAS @ NOP':
                            logos_extracted_3.append('LEAGUE PASS')
                            desc_extracted_3.append('GSW @ DAL')

            week_num +=1
        logos_extracted_total += logos_extracted_3
        desc_extracted_total += desc_extracted_3
        print("games done so far: " + str(len(logos_extracted_total)))
        for desc, link in zip(desc_extracted_3, file):
            assert desc.lower().split()[0] in link and desc.lower().split()[
                2] in link, "link: " + link + ' content: ' + desc + "all content: " + ' '.join(desc_extracted_3)
        print("3/3 order verified")

    driver.quit()
    print("finished parsing \n")
    print("Finished, total number of elements: " + str(len(logos_extracted_total)))
    print("\nunique broadcasters: ")
    print(set(logos_extracted_total))
    print("\n verifying order...")
    print("OK!")
    #creating list to write
    broadcast_list = []
    for bc, desc in zip(logos_extracted_total, desc_extracted_total):
        broadcast_list.append([bc,''.join(desc)])
    with open('../full season data/broadcast_list.txt', mode='w') as file:
        json.dump(broadcast_list,file)
    print("saved bc_list")
    bc_types = list(Counter(logos_extracted_total).items())
    with open('../full season data/unique_broadcasts.txt', mode='w') as file:
        json.dump(bc_types,file)
    print("saved unique_bc")


if __name__ == '__main__':
    main()