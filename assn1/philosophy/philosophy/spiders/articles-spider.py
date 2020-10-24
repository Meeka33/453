import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from philosophy.items import WebfocusedcrawlItem
import os.path
from nltk.corpus import stopwords


def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    good_tokens = [token for token in tokens if token not in stopword_list]
    return good_tokens

#comb through first and second level links of stanford encyclopedia of philosophy extracting text:
class ArticlesSpider(CrawlSpider):
    name='articles-spider'
    custom_settings={
        'DEPTH_LIMIT': '1'
    }
    allowed_domains=['plato.stanford.edu']
    start_urls=['https://plato.stanford.edu/entries/wittgenstein/']
    rules=[
        Rule(LinkExtractor(restrict_xpaths='//div[@id="related-entries"]'), callback='parse_start_url', follow=True),
        ]

    def parse_start_url(self, response):
        # first part: save individual page html to philosophy directory
        page = response.url.split("/")[4]
        page_dirname = 'philanguage'
        filename = '%s.html' % page
        with open(os.path.join(page_dirname,filename), 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

        # Second extract text for corpus creation
        item=WebfocusedcrawlItem()
        item['url']=response.url
        item['title']=response.css('h1::text').extract_first()

        bodytext=[]
        divs=response.xpath('//div[@id=''"main-text"]')
        for p in divs.xpath('*[not(self::h2) and not(self::h3)]'):
            bodytext.append(p.get())
        item['text']=bodytext

        tags_list = [response.url.split("/")[2], response.url.split("/")[4]]
        item['tags'] = tags_list
        return item
