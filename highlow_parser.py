from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

def get_play_component_text(page_url):
    # Initialize WebDriver (make sure to specify the path to your WebDriver)
    driver = webdriver.Chrome()

    # Open the page URL
    driver.get(page_url)

    # Wait for the content to load
    time.sleep(10)  # Adjust the sleep time as needed

    # Get the page source after the dynamic content is loaded
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Close the WebDriver
    driver.quit()

    # Find all elements with the class 'play-component'
    play_components = soup.find_all(class_='play-component')

    if not play_components:
        print("No play components found. Verify the class name or structure of the HTML.")
    else:
        print(f"Found {len(play_components)} play components.")

    # Extract the text from each 'play-component'
    play_texts = [component.get_text(strip=True) for component in play_components]

    return play_texts

def save_to_file(text_list, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for text in text_list:
            file.write(f"{text}\n")
            file.write('---\n')

def main():
    page_url = 'https://thehighlow.io/video/ids?ids=24zcKM24q5aY24bGEy24tNKv24znTy245wjc2422PS240eO924LaVr24Evjv24TxN424fZMd24YP3t24f2Fh24PQeO24jVPE24mN5P24S3Eg24KYEG24WCJs24REEp240HCk24Y7E124hovc24AxfI24wEZ024fjZw24y1jq24wlR824jdmx24A8LZ24IoXX24gR2224ypj124nqZz24FHA024D9aL24uGwb24HBxc24c2tk24aoj224Rw3Z24VPE324UhhD246IzE24C6bL24vatm24yf0S24CWsC249XFU24ae5q24kzyL24EJP924aHs824hbxC24NyoF24DsHD24BfI724SyH724R9Ig241Fae24JHVO24q7he24D2pk240tvU24yGu12490Yu244Q5024RScb24Y5jn24JG7L24Lqa424N8dV24KINL242J8N24lERM24ma6P24tl0X24qVSK24g9W724WVGx24b7qs24rPio24ffFF24uYOG24GLai244kOV24W1x024OIHW24SpN924ro0Y24JqgS24uNoV24uTDw240ERK24LGbC248uJg24BjK424dfx6246t9X24VJoG24NVFH24OQiu24LiKE24v5z924j8jU24QBKZ24WP4724jF3q24BHR3245UN424IfVi244yDU247jdA24i1NU24n0mV24mLX824MCkQ24G3IA24NUhi24oRws24J6fU24LriJ243RN724nIqT24BSMD24dWGV24H2Fc24mURE24egB624VTbG24UZdY24ScWl248Dy324HrFY24JMIq24bsWa24vqx0244N0N24kZj1240XPy24Zfsa24OEgy24875f24L3f824e2Tg24fMSA243uR124EyZP24oO6624bFtv24TIJW243q8M24f3oV24DI0k24aGiU24cmwN24wm9l24qcrk24Xzyb24PxRw24jo3X24BqQq24Pna224ilZu24iqXT24SCVh2415Vi24nPi224dxtU24UvDO24ANjl24iLmZ247X1e24qXjX243rZX24HA1m24I2eJ24LMk024lt4v243cKl24jidV246HQ424CNPB24yWuJ24nvLj249EeE24blw224bStE247Jpd24qwyA24nH1C24ypTr24cQAk24jowi24cWlf24jT1t24WllO2424OQ248A9B240r4c2481Cj24TBO024Se3r24MP3J240PYR24JM7524llct24PoIh24BZ2224cqBf24SDwX24mNmc24P2Ig24FZMB24uh7G24j5Md24QKoz24eE5G24dFiD24MtVh24j6NS24bSOw24hOQs24Mdfe24QmNE245LeC24RTS324qmTL24M0Io24jKkj24tCjJ24RkJA'
    play_texts = get_play_component_text(page_url)

    save_to_file(play_texts, 'play_components.txt')
    print(f"Play component text saved to play_components.txt")

if __name__ == '__main__':
    main()
