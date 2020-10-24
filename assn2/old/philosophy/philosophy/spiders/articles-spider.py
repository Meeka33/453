import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from philosophy.items import WebfocusedcrawlItem
import string
import re

#Directories for HTML, text files:
html_directory = '/home/meeka/Desktop/NU/453/assn2/philosophy/philanguage'
corpus_directory = '/home/meeka/Desktop/NU/453/assn2/philosophy/corpus'

#comb through first and second level links of stanford encyclopedia of philosophy extracting text:
class ArticlesSpider(CrawlSpider):
    name='articles-spider'
    custom_settings={
        'DEPTH_LIMIT': '2'
    }
    allowed_domains=['plato.stanford.edu']
    start_urls=['https://plato.stanford.edu/entries/wittgenstein/']
    rules=[
        Rule(LinkExtractor(restrict_xpaths='//div[@id="related-entries"]'), callback='parse_start_url', follow=True),
        ]

#Functions for HTML extraction, cleaning, text creation
#######################################################

def parse_start_url(self, response):
    # first part: save individual page html to philosophy directory
    page = response.url.split("/")[4]
    filename = '%s.html' % page
    with open(os.path.join(html_directory,filename), 'wb') as f:
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
    #extract tags
    tags_list = [response.url.split("/")[2], response.url.split("/")[4]]
    item['tags'] = tags_list
    return item

# HTML Parsing, save to TXT

def save_txt(lines, filename):
    file=open(filename, 'w')
    file.write(lines)
    file.close()

def process_htmldocs(directory):
    for filename in os.listdir(directory):
        path=directory + '/' + filename
        load_htmldoc(path)

def load_htmldoc(filename):
    file=open(filename, 'r')
    content=file.read()
    file.close()

    #Main text extraction
    webtext=BeautifulSoup(content, 'html.parser')
    pagetext=webtext.find(id="main-text")

    #Clean subject header text
    subject=[headers.text.encode('utf-8').decode('ascii', 'ignore') for headers in webtext.find_all('h1', limit=1)]
    char_remove=re.compile('[%s]' % re.escape(string.punctuation))
    subject=[char_remove.sub(' ', w) for w in subject]

    #Cycle through each entry to split sub-sections into unique docs
    topics=[]
    for headers in pagetext.find_all('h2'):
        headers=headers.text.encode('utf-8').decode('ascii', 'ignore')
        topics.append(subject[0]+'_'+headers)
    topics=[char_remove.sub(' ', w) for w in topics]

    #Split HTML sections by header classes
    collection=[]
    body_split = re.split('(<h2>|</h2>)', str(pagetext))
    for section in body_split:
        clean=BeautifulSoup(section, 'html.parser')
        paragraphs=clean.select('p')
        if paragraphs:
            collection.append(paragraphs)

    #Clean individual sections, remove remaining tags
    counter=0
    for section in collection:
        text=''
        htmlstrip=['\n','<p>','</p>','<em>','</em>']
        for x in range(len(section)):
            sent=str(section[x].text)
            for item in htmlstrip:
                sent=sent.strip().replace(item, ' ')
            text=text +' '+ sent
        textfile = corpus_directory + '/' + topics[counter]+'.txt'
        save_txt(text, textfile)
        counter=counter+1


process_htmldocs(html_directory)
