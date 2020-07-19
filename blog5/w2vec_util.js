function clean(text){

    let textt = text.split(' ')

    //filter out empty string
    let text_filter = textt.filter((val)=>{
            return val !=''
    });

    let stop_wordFilter = text_filter.filter((val)=>{
            
            return !stopwords.includes(val);
    });

    let puntionless = stop_wordFilter.map((val)=>{

         return val.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"")
    });

    return puntionless;
}

function gen_word(window=5, text_list){

    let word_list = []
    let all_text = []

    for(let index in text_list){

        let text = text_list[index];

        let text_clean = clean(text);

        all_text.push(...text_clean);


        for(let i=0;i <text_clean.length; i++){

            let word = text_clean[i]
            for(let j=0;j < window; j++){

                if((i + 1 + j) < text_clean.length){
                    word_list.push([word, text_clean[i+1+j]])
                }

                if((i-j-1) >=0){
                    word_list.push([word, text_clean[i-j-1]])
                }
            }
        }
        
    }

    return [word_list, all_text]
}

function unique_word(text_list){

    let word_set = new Set(text_list);

    let word_list = Array.from(word_set);

    let unique_word  = {}

    for(let i=0; i < word_list.length; i++){
        
         let word = word_list[i]
         unique_word[word] = i+1;
    }

    return unique_word
}

function obj_len(dict){

    let count = 0

    for(let i in dict){
        count +=1
    }
    return count;
}

function create_data(word_list){

    let  data = []
    let label = []

    for(let i=0; i< word_list.length;i++){

        let x = word_list[i][0]
        let y = word_list[i][1]

        let word_index = unique_dict[x]
        let context_index = unique_dict[y]

        let X_row = utils.zeros(n_words)
        // let y_row = utils.zeros(n_words)

        X_row[word_index] = 1.

        // y_row[word_index] = 1.

        data.push(X_row)
        label.push(context_index)
    }

    return [data, label];
}

function get_weight(weight){

    let row = weight.n

    let col = weight.d

    let data = Array(row);

    for(var i=0;i<row;i++){
        data[i] =[]
        for(var j=0;j<col;j++){
            /**
             * since the array are store in this form [1,2,3,4,5,6,7,8,9,10,11,12]
             * such an array suppose to be a 2d array [[1,2,3,4,5,6],[7,8,9,10,11,12]]
             * in which the first array in the 2d array is 0
             * the best way to specifiy the index in the 1d array is:
             * (arr.length*jth_col).column.length + ith_row
             */
            var indices = (weight.out.length*j)/col + i;
            
            data[i].push(weight.out[indices]);
        }
    }
    
    return data


}
