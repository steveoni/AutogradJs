
let text = "The king is a man who rules over a nation, he always have a woman beside him called the\
 queen.\n she helps the king controls the affars of the nation.\n Perhaps, she acclaimed the position of a king\
 when the king her husband is deceased."


text_lower = text.toLocaleLowerCase()

text_list = text_lower.split("\n")

var stopwords = ["a","in","when","the","of","is","who"]


let [word_list, all_text] = gen_word(5,text_list);


let unique_dict = unique_word(all_text)

let n_words = obj_len(unique_dict);

console.log(n_words);


let [data, label] = create_data(word_list)


let embed_dim = 50;
let model = new nn.Sequential([
        new nn.Linear(n_words,embed_dim),
        new nn.Linear(embed_dim,n_words),
        new nn.Softmax()
]);

let optim = new nn.OptimSGD(model,lr=0.001);

//Train the model.

epoch = 50
for(let i=0; i< epoch; i++){

    let total_loss = 0;
    for(let j=0; j < data.length; j++){

        let x_data = data[j]
        let y_data = label[j]

        let x = new nn.Tensor(1,n_words, false);
        x.setFrom(x_data)

        model.forward(x)

        // console.log(-Math.log(model.out.out[y_data-1]))
        let loss = new nn.Loss(y_data-1,model)

        // console.log(loss.out);
        total_loss += loss.out

        loss.backward()

        optim.step();

        optim.grad_zero()

    }

    console.log(`for epoch ${i} Loss is ${total_loss/data.length}`)
}

//get embedding weight

let embed_weight = get_weight(model.models[0].W)
console.log(embed_weight[0].length)
