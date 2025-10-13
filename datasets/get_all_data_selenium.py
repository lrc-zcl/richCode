"""
爬取体彩所有历史数据 selenium
"""
import time
import pandas as pd
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class CrawlSelenium():
    def __init__(self, base_url):
        self.header = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 Edg/135.0.0.0"
        }
        self.base_url = base_url
        self.browser = webdriver.Edge()

        self.browser.get(self.base_url)

    def get_signal_page_data(self):

        self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        self.browser.execute_script("window.scrollTo(0, 0);")
        iframe = WebDriverWait(self.browser, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "iframe"))
        )

        # 切换到 iframe
        self.browser.switch_to.frame(iframe)
        url_data = self.browser.page_source
        etree_html = etree.HTML(url_data)
        current_page_history_data = etree_html.xpath("/html/body/div/div/table/tbody/tr")
        current_page_all_data = []
        for signal_date_data in current_page_history_data:
            all_td_data = signal_date_data.xpath("./td")  # 所有的td 标签
            signal_write_deta_data = []
            for index, signal_td_data in enumerate(all_td_data[0:9]):
                if index == 1:
                    data = signal_td_data.xpath("./text()")[0]
                else:
                    data = int(signal_td_data.xpath("./text()")[0])
                signal_write_deta_data.append(data)
            current_page_all_data.append(signal_write_deta_data)
        return current_page_all_data

    def get_all_page_data(self, page_counts=140):
        all_write_data = []
        for s in range(page_counts):
            signal_page_data = self.get_signal_page_data()
            all_write_data.extend(signal_page_data)

            button = WebDriverWait(self.browser, 3).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div/div/div[2]/ul/li[13]")))  # 点击下一页
            button.click()
            # 切换回主页面
            self.browser.switch_to.default_content()
            print(f"目前是第{s + 1}页")
        return all_write_data


if __name__ == "__main__":
    base_url = "https://www.lottery.gov.cn/kj/kjlb.html?js7"
    crawl_request = CrawlSelenium(base_url)
    all_write_data = crawl_request.get_all_page_data()
    columes = ["期号", "开奖日期", "数字1", "数字2", "数字3", "数字4", "数字5", "数字6", "数字7"]
    df = pd.DataFrame(all_write_data, columns=columes)
    df.to_excel("../data/lottery_dataiii.xlsx", index=False)
    print("excel写入完成!!")
