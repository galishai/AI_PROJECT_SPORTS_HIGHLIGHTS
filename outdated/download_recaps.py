import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import subprocess

chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument('capabilities={"acceptInsecureCerts": True}')
chrome_options.add_argument("--enable-logging")

import urllib.request
urllib.request.urlretrieve("https://nbalpng.akamaized.net/vod/hls-itc/NBA_202404141910NBA_____VIDEOS__NBAE_2741789/v0_121-181685-t-1279278.m4s?hdnts=exp%3D1732310162~acl%3D%2F*~hmac%3Dcda95fc3734b1f65f9befe819e55bf427c9c439d7394238425fd3510c3f8c526", 'video_name.mp4')

def capture_requests(driver):
    # Enable network tracking
    driver.execute_cdp_cmd("Network.enable", {})
    requests = []

    # Define a listener for network events
    def intercept_request(event):
        if "params" in event and "request" in event["params"]:
            requests.append(event["params"]["request"]["url"])

    # Use Chrome's DevTools logging to extract requests
    log_entries = driver.get_log("performance")
    for entry in log_entries:
        log_data = entry["message"]
        if "Network.requestWillBeSent" in log_data:
            intercept_request(log_data)

    return requests


def download_video_with_ytdlp(page_url):
    print("Starting video download with yt-dlp...")
    subprocess.run(["yt-dlp", page_url])


# Get the video URL from the page
def get_video_url(page_url):
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(page_url)

    # Wait for the page to load completely
    driver.implicitly_wait(10)

    # Extract the page source
    page_source = driver.page_source
    driver.quit()

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(page_source, "html.parser")

    # Locate the video tag or the link to the video
    video_element = soup.find("video")
    if video_element and video_element.get("src"):
        return video_element["src"]

    # Fallback: Check for embedded links in script tags
    script_tags = soup.find_all("script")
    for script in script_tags:
        if "recap" in script.text and "mp4" in script.text:
            # Extract the video URL
            start_index = script.text.find("http")
            end_index = script.text.find(".mp4") + 4
            return script.text[start_index:end_index]

    return None


def get_video_source(page_url):
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(page_url)
    # Wait for the page to load fully
    time.sleep(5)

    # Retrieve the video element and its `src`
    try:
        wait = WebDriverWait(driver, 20)

        reference_elements = driver.find_elements(By.CSS_SELECTOR, 'svg[width="40"][height="40"][viewBox="0 0 20 20"]')
        assert (len(reference_elements) == 1, "Number of elements: " + str(len(reference_elements)))
        actions = ActionChains(driver)
        actions.move_to_element_with_offset(reference_elements[0], 0, 0).click().perform()
        requests = capture_requests(driver)
        media_requests = [req for req in requests if any(ext in req for ext in [".mp4", ".m3u8"])]
        print("Captured media requests:", media_requests)
        '''
        for request in driver.requests:
            if request.response and ("mp4" in request.url or "m3u8" in request.url):
                print(f"Video URL found: {request.url}")
                break
        '''

        print("Clicked at coordinates")

        # Click the play button to start the video
        #play_button.click()
        #print("Clicked the play button!")

        # Allow time for the video to start playing and load the `src`
        time.sleep(3)  # Adjust this delay as needed
        current_url = driver.current_url
        if current_url:
            download_video_with_ytdlp(current_url)
        driver.quit()
        '''
        for request in driver.requests:
            if request.response and ("mp4" in request.url or "m3u8" in request.url):
                print(f"Video URL found: {request.url}")
                break
        '''
        video_element = driver.find_element(By.CLASS_NAME, "video-player")
        video_src = video_element.get_attribute("src")
        if "blob:" in video_src:

            print("Video uses blob URL. Inspecting network requests...")
            script = f"""
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '{video_src}', true);
            xhr.responseType = 'blob';
            xhr.onload = function() {{
                console.log(URL.createObjectURL(xhr.response));
            }};
            xhr.send();
            """
            driver.execute_script(script)
            return None
        else:
            return video_src
    finally:
        driver.quit()


# Download the video
def download_video(video_url, output_file):
    response = requests.get(video_url, stream=True)
    response.raise_for_status()

    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Downloaded video to {output_file}")


# Main script
def main():
    page_url = "https://www.nba.com/game/uta-vs-gsw-0022301198/box-score%23box-score?watchRecap=true"
    output_file = "game_recap.mp4"

    print("Retrieving video source...")
    video_src = get_video_source(page_url)

    if video_src:
        print(f"Video source found: {video_src}")
        print("Downloading video...")
        download_video(video_src, output_file)
    else:
        print("Could not retrieve the video source. Check network requests manually.")


if __name__ == "__main__":
    main()
