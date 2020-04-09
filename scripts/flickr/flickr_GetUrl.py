#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# https://gist.github.com/Gunnvant/edd0754a79956a699f4f2cf05fa7e42c#file-flickr_geturl-py
## run
## > python flickr_GetUrl.py tag number_of_images_to_attempt_to_download

#Flickr API extras extras
# http://librdf.org/flickcurl/api/flickcurl-searching-search-extras.html
from flickrapi import FlickrAPI
import pandas as pd
import sys
key='f6a182f0dc4be83ec7b4d1557e710f29'
secret='2535ebafcf2ec5db'

def get_urls(image_tag,MAX_COUNT,mode='any',url_type='url_o',per_page=50):
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(text=image_tag,
                            tag_mode=mode,
                            tags=image_tag,
                            extras=url_type,
                            per_page=50,
                            sort='relevance')
    count=0
    urls=[]
    for photo in photos:
        if count< MAX_COUNT:
            count=count+1
            print("Fetching url for image number {}".format(count))
            try:
                url=photo.get('url_o')
                urls.append(url)
            except:
                print("Url for image number {} could not be fetched".format(count))
        else:
            print("Done fetching urls, fetched {} urls out of {}".format(len(urls),MAX_COUNT))
            break
    urls=pd.Series(urls)
    print("Writing out the urls in the current directory")
    urls.to_csv(image_tag+"_urls.csv")
    print("Done!!!")
def main():
    tag=sys.argv[1]
    MAX_COUNT=int(sys.argv[2])
    tagmode=sys.argv[3]
    urltype=sys.argv[4]
    perpage=sys.argv[5]
    get_urls(tag,MAX_COUNT,tagmode,urltype,perpage)
if __name__=='__main__':
    main()
