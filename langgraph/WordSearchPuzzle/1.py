
    #def get_words_to_search() -> str:
    #def get_of_the_marvel_heros_on_image() -> str:   # get  list of marvel heros. names must be in uppercase & json format. remove spaces from names
    
    #def trav_to_Search_words(self) -> str: 
        
        #for right
        #for left
        #for down
        #for up
        #for primDiag
        #for secDiag
        #for revere     
       # mark hero-name, start-co-ods, cell-co-ods[], direction

    def PerformWordSearch(self, wordsList:list[str], grid) -> ResultWordSearchPuzzle:
        wordListLength=len(wordsList)
        gridRows=len(grid)
        gridCols=len(grid[0])
        
        logger.debug(f"grid W x H= {gridCols} x {gridRows}: {grid}")
        result=ResultWordSearchPuzzle(wordResults=None)
        # result.wordResults[0].
        for rowTrav in range(gridRows):
            forwardString="".join(grid[rowTrav])
             
            for wordTrav in range(wordListLength):
                wordN=wordsList[wordTrav]
                if((result.wordResults is not None)  and any(getattr(obj, "word") == wordN for obj in result)  ):
                    continue
                
                wordResult=None
                #horizontal L to R
                wIndex=self.GetIndexOfWordSubstring(forwardString)
                if(wIndex > 0):
                    wordResult=WordResult ( word=  wordN,startIndex=wIndex,
                                           direction="horizontal-L-to-R")
                    result.wordResults.append(wordResult)
                    continue
                #horizontal R to L
                reverseString=forwardString[::-1]
                wIndex=self.GetIndexOfWordSubstring(forwardString)
                if(wIndex > 0):
                    wordResult=WordResult ( word=  wordN,startIndex=wIndex,
                                           direction="horizontal-R-to-L")
                    result.wordResults.append(wordResult)
                    continue
        #vertical T-to B
        transposed_columns = itertools.zip_longest(*grid, fillvalue='')
        logger.debug(f"transposed_columns:   {transposed_columns}")        
        #vertical_strings_padded = ["".join(col).rstrip() for col in transposed_columns]
        for colTrav in range(gridCols):
            forwardString=transposed_columns[colTrav]
            for wordTrav in range(wordListLength):

                wordN=wordsList[wordTrav]
                if((result.wordResults is not None)  and any(getattr(obj, "word") == wordN for obj in result)  ):
                    continue
                    
                wordResult=None
                #horizontal L to R
                wIndex=self.GetIndexOfWordSubstring(forwardString)
                if(wIndex > 0):
                    wordResult=WordResult ( word=  wordN,startIndex=wIndex,
                                            direction="horizontal-L-to-R")
                    result.wordResults.append(wordResult)
                    continue
                #horizontal R to L
                reverseString=forwardString[::-1]
                wIndex=self.GetIndexOfWordSubstring(forwardString)
                if(wIndex > 0):
                    wordResult=WordResult ( word=  wordN,startIndex=wIndex,
                                            direction="horizontal-R-to-L")
                    result.wordResults.append(wordResult)
                    continue
        return result

    def GetIndexOfWordSubstring(self, word:str, row:str)-> int:
        try:
            return row.index(word)
        except ValueError as e:
            return 0  
# Example usage
#my_char_list = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
#search_sequence_list = ['w', 'o', 'r', 'l', 'd']
#if search_sequence_in_list(my_char_list, search_sequence_list):
 # print(f" {True}")
 


    def create_htmlOutput() -> str:
        html=""        
        #loop thru the char array 


    def Get_CharArray_Prompt(self, url: str) -> str:
        prompt = f""" 
            Extract the characters from within the grid in image and return as json array 
            User Input: image url {url} """
        return prompt
    
    def Get_wordsOutsideGrid_Prompt(self, url: str) -> str:
        prompt = f""" 
            Extract the words from outside the grid in image and return as json array 
            User Input: image url {url} """
        return prompt
        

        


    def Get_LLM_response(self, prompt: str) -> str:
        """ LLM """
        try:
            config = load_config()
            llm = ChatGroq(api_key=config.groq_api_key, model_name="llama-3.1-8b-instant")
            logger.debug(f"LLM prompt formed:  {prompt}")
            response = llm.invoke(prompt)
            resp = response.content.strip()
            logger.debug(f"response : {resp}")
            return resp 
            
        except Exception as e:
            logger.error(f"Error in extracting char array from image: {str(e)}")
            return None
        
    def extract_CharArray_from_ImageUrl(self, base64image: str) -> str:
        """Extract topic for podcast creation using LLM"""
        try:
            config = load_config()
            llm = ChatGroq(api_key=config.groq_api_key, model_name="llama-3.1-8b-instant")
            
            GetCharsFromGrid_prompt = f"""
                Extract the characters from the grid in image and return as json array 
                User Input: image url {base64image}
                """            
            logger.debug(f"LLM prompt formed:  {GetCharsFromGrid_prompt}")             
            resp =self.Get_LLM_response( GetCharsFromGrid_prompt) 

            logger.error(f"CharArray : {resp}")
            return resp #if topic else "general discussion"
            
        except Exception as e:
            logger.error(f"Error in extracting char array from image: {str(e)}")
            # Fallback  
           
            return None
                
    def Get_base64Str_from_localImage(self, imgPath: str) -> str:
        """invoke LLM and get the char array"""
        try:        
            logger.debug(f"image path: {imgPath}")
            base64Str=ImageHelper.image_to_base64_string(imgPath)
            logger.info(f"Successfully read image as string: Str-Length {len(base64Str)}  ") 
            return base64Str
                    
        except Exception as e:
            logger.error(f"Error in Get_base64Str_from_localImage : {str(e)}")
            return None