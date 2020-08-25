const utils = require("./utils.js")
const Network = require("./model_graph.js")
const word_utils = require("./w2vec_util.js")

var autograd = {};

(function(global){

    function name(arg){

        if("name" in arg){
            arg.name += utils.randi(1,6)
        }else{
            arg.func_name += utils.randi(1,6)
        }
    }

    function Scalar(arr,require_grad){

        this.item = arr;
        this.require_grad = require_grad;
        this.gradv = 0;
        this.name = "<Scalar>";
    
    }
    
    Scalar.prototype = {
    
        grad: function(g){
            this.gradv = g;
        }
    }

    function add(x,y){


        this.x = x;
        this.y = y;
        name(x)
        name(y)
        this.require_grad=true;
    
        this.item = x.item + y.item
        this.gradv = 0;
        this.func_name = "<scalar_Add>";
    
    }
    
    add.prototype = {
    
        backward: function(){
    
            if(this.x.require_grad){
                    console.log( this.x instanceof Tensor, this.x.item )
                    this.x.grad(1*this.gradv);
                    if("backward" in this.x){
                        this.x.backward()
                    }
    
    
            }
    
            if(this.y.require_grad){
                this.y.grad(1*this.gradv);
                if("backward" in this.y){
                    this.y.backward()
                }
            }
        },
    
        grad: function(g){
    
            this.gradv = g;
    
        }
    }
    
    
    
    function mult(x, y) {
    
        this.item = x.item * y.item;
        this.x = x;
        this.y = y;
    
        name(x)
        name(y)
        this.gradv = 0;
    
        this.require_grad = true;
        this.func_name = "<scalar_Multi>"
    
    }
    
    mult.prototype = {
    
        backward: function () {
    
            if (this.x.require_grad) {
                // console.log( this.x instanceof Tensor, this.x.item )
                this.x.grad(this.y.item * this.gradv);
                if ("backward" in this.x) {
                    // console.log("True")
                    this.x.backward()
                }
    
    
            }
    
            if (this.y.require_grad) {
                this.y.grad(this.x.item * this.gradv);
                if ("backward" in this.y) {
                    this.y.backward()
                }
            }
        },
    
        grad: function (g) {
            this.gradv =  g;
        }
    }

    let Mat = function (n, d) {
        // n is number of rows d is number of columns
        this.n = n;
        this.d = d;
        this.out = utils.zeros(n * d);
        this.dout = utils.zeros(n * d);
    }

    let Tensor = function (n, d, require_grad) {

        this.n = n;
        this.d = d;
        this.out = utils.zeros(n * d);
        this.dout = utils.zeros(n * d);
        this.require_grad = require_grad;
        this.name = null;
    }

    Tensor.prototype = {

        get: function (row, col) {
    
            var ix = (this.d * row) + col;
            utils.assert(ix >= 0 && ix < this.out.length);
            return this.out[ix];
    
        },
        set: function (row, col, v) {
            var ix = (this.d * row) + col;
            utils.assert(ix >= 0 && ix < this.out.length);
            this.out[ix] = v;
        },
        setFrom: function (arr) {

            if(Array.isArray(arr[0])){

                let n = arr.length;
                let d = arr[0].length;
                utils.assert(n*d == this.n*this.d,"shape not compatible")

                let out = []

                for(let i=0; i < arr.length; i++){
                    let row = arr[i]
                    out.push(...row)
                }
                this.out = out
                this.out_tensor = arr
            }else{
                utils.assert(arr.length == this.n*this.d,"shape not compatible")
                for (var i = 0, n = arr.length; i < n; i++) {
                    this.out[i] = arr[i];
                }
            }
            
        },
        randn: function (mu, std) {
            utils.fillRandn(this.out, mu, std);
            return this;
    
        },
        grad: function (grad) {
    
            // for(var i=0,n=grad.length;i<n;i++){
            //     this.dout[i] = grad[i];
            // }
            this.dout = grad;
        }
    }

    function addTensor(x, y) {
        utils.assert(x.out.length === y.out.length);
    
    
        this.items = new Mat(1,x.out.length);
        for (let i = 0; i < x.out.length; i++) {
    
            this.items.out[i] = x.out[i] + y.out[i];
        }
        this.x = x;
        this.y = y;
        this.require_grad = true;
        this.out = this.items.out;
        this.dout = this.items.dout;
        this.n = this.items.n;
        this.d = this.items.d;
        this.func_name = "<add>";
    
        // this.gradv = 1;
    
    }

    addTensor.prototype = {

        backward: function () {
    
            if (this.x.require_grad) {
                this.x.grad(this.dout);
                if ("backward" in this.x) {
                    this.x.backward()
                }
    
    
            }
    
            if (this.y.require_grad) {
                this.y.grad(this.dout);
                if ("backward" in this.y) {
                    this.y.backward()
                }
            }
        },
    
        grad: function (g) {
    
            utils.assert(this.items.dout.length === g.length);
            this.dout = g;
    
        }
    }

    function Matmul(x, y) {

        utils.assert(x.d === y.n, "matmul dimension misaligned");
    
        this.n = x.n;
        this.d = y.d;
        this.x = x;
        this.y = y;
        this.require_grad = true;
        this.items = new Mat(this.n, this.d);
        this.out = this.items.out;
        this.dout = this.items.dout
        this.func_name = "<Multiply>";
    
        for (let i = 0; i < x.n; i++) {
            for (let j = 0; j < y.d; j++) {
    
                var dot = 0.0;
                for (var k = 0; k < x.d; k++) {
    
                    dot += this.x.out[x.d * i + k] * this.y.out[y.d * k + j];
                }
                this.out[this.d * i + j] = dot;
            }
        }
    }

    Matmul.prototype = {

        backward: function () {
    
            if (this.x.require_grad) {
                
                for(let i = 0;i< this.x.n;i++){
                    for(let j=0;j<this.y.d;j++){
                        for(let k =0;k<this.x.d;k++){
                            let b = this.dout[this.y.d*i+j];
                            // console.log(b);
                            this.x.dout[this.x.d*i+k] += this.y.out[this.y.d*k+j] * b;
    
                        }
                    }
                }
    
                if ("backward" in this.x) {
                    this.x.backward()
                }
    
    
            }
    
            if (this.y.require_grad) {
    
                for(let i = 0;i< this.x.n;i++){
                    for(let j=0;j<this.y.d;j++){
                        for(let k =0;k<this.x.d;k++){
                            let b = this.dout[this.y.d*i+j];
                            this.y.dout[this.y.d*k+j] += this.x.out[this.x.d*i+k] * b;
    
                        }
                    }
                }      
    
                if ("backward" in this.y) {
                    this.y.backward()
                }
            }
        },
    
        grad: function (g) {
    
            utils.assert(this.dout.length === g.length);
            this.dout = g;
    
        }
    }

    function Linear(in_dim,out_dim){

        this.W = new Tensor(in_dim,out_dim,true).randn(0,0.008);
        this.b = new Tensor(out_dim,1,true).randn(0,0.008);
    
        this.func_name = "<Linear>";
        this.require_grad = true;
    
    }

    Linear.prototype = {

        forward : function(x){
    
            this.mult = new Matmul(x,this.W)
            this.items = new addTensor(this.mult,this.b);
            this.n = this.mult.n;
            this.d = this.mult.d;
            this.out = this.items.out;
            this.dout = this.items.dout;
            
    
            return this;
        },
    
        backward: function(){
    
                this.items.grad(this.dout);
                this.items.backward();
        },
        grad: function(g){
            utils.assert(this.dout.length === g.length);
            this.dout = g;
        },
    
        update: function(lr){
    
            for(var i=0;i< this.W.out.length;i++){
    
                this.W.out[i] -= (lr*this.W.dout[i]); 
            }
            for(var i=0;i< this.b.out.length;i++){
    
                this.b.out[i] -= (lr*this.b.dout[i]); 
            }
    
    
    
        },
        get_weight: function(){

            let row = this.W.n

            let col = this.W.d

            let data = Array(row);

            for(let i=0;i<row;i++){
                data[i] =[]
                for(let j=0;j<col;j++){
                    /**
                     * since the array are store in this form [1,2,3,4,5,6,7,8,9,10,11,12]
                     * such an array suppose to be a 2d array [[1,2,3,4,5,6],[7,8,9,10,11,12]]
                     * in which the first array in the 2d array is 0
                     * the best way to specifiy the index in the 1d array is:
                     * (arr.length*jth_col).column.length + ith_row
                     */
                    let indices = (this.W.out.length*j)/col + i;
                    
                    data[i].push(this.W.out[indices]);
                }
            }
            
            return data
        }

    }
    
    function ReLU(){

        this.require_grad = true;
        this.func_name = "<ReLu>";
    
    }

    ReLU.prototype = {

        forward : function(x){
            this.items = new Mat(x.n,x.d);
            this.x = x;
            this.n = x.n;
            this.d = x.d;
            
            for(var i=0;i<x.out.length;i++){
    
                this.items.out[i] = Math.max(0,x.out[i]);
            }
            this.out = this.items.out;
            this.dout = this.items.dout;
    
            return this;
        },
    
        backward: function(){
            
            
            for(var i=0;i<this.x.out.length;i++){
    
                this.x.dout[i] = this.x.out[i] > 0 ? this.dout[i] : 0.0;
            }
            this.x.backward();
    
    
        },
        grad: function(g){
            utils.assert(this.dout.length === g.length);
    
            this.dout = g;
    
        }
    }

    function Softmax(){
        this.func_name = "<Softmax>";
    }

    Softmax.prototype = {

        forward : function(x){
            this.items = new Mat(1,x.d)
    
            this.x = x;
            //compute max activation
            var as = x.out;
            var amax = x.out[0];
            for(var i=1;i<this.x.d;i++){
    
                if(as[i] > amax) amax = as[i];
            }
    
            var es = utils.zeros(this.x.d);
            var esum = 0.0;
            for(var i=0;i<this.x.d;i++){
    
                var e = Math.exp(as[i] - amax);
                esum += e;
                es[i] = e;
            }
    
            //normalize and output to sum one
            for(var i=0;i<this.x.d;i++){
                es[i] /= esum;
                this.items.out[i] = es[i];
            }
            this.out = es; // saved for backprop
    
            return this;
        },
    
        backward: function(y){
    
            var x = this.x;
            for(var i=0;i<this.items.d;i++){
    
                var indicator = i === y ? 1.0 : 0.0;
                var mul = -(indicator - this.out[i])
                x.dout[i] = mul;
            }
    
            this.x.grad(x.dout)//there is no need for this. we initiallizing dw twice
            this.x.backward();
        }
    }

    function Sequential(models){

        this.models = models;

        this.weights = []

        let w_count = 0;
        for(let i=0;i<this.models.length;i++){

            this.models[i].func_name = `${this.models[i].func_name}_${i}`

            if(this.models[i].hasOwnProperty("W") && this.models[i].hasOwnProperty("b")){
                
                this.models[i].W.name = `weight_${w_count}`
                this.models[i].b.name = `bias_${w_count}`

                this.weights.push(this.models[i].W.name)
                this.weights.push(this.models[i].b.name)

                w_count +=1
            }

        }
    }

    Sequential.prototype = {

        forward : function(x){
    
            
            this.out = this.models[0].forward(x);
            

            for(var i=0;i<this.models.length;i++){
    
                if( i==0){
                    continue;
                }
                this.out = this.models[i].forward(this.out);
                if(this.out.hasOwnProperty("mult") && this.out.hasOwnProperty("items")){
                    this.out.mult.func_name = `${this.out.mult.func_name}_${i}`;
                    this.out.items.func_name = `${this.out.items.func_name}_${i}`;
                }
            }
        },
    
    }

    function Loss(target, predict){

        this.model = predict.out;
        this.out =  - Math.log(predict.out.out[target]);;
        this.y = target;
    }

    Loss.prototype = {

        backward : function(){
    
            this.model.backward(this.y);
            
        }
    }

    function OptimSGD(model,lr){

        this.model = model;
        this.lr = lr;
    
    }
    
    OptimSGD.prototype = {

        step : function(){
    
            for(let i in this.model.models){
                
                // console.log(model)
                if("update" in this.model.models[i]){
                    // console.log("here")
                    this.model.models[i].update(this.lr);
                }
            }
        },
        grad_zero:  function(){

            for(var i in this.model.models){
                
                // console.log(model)
                if(Object.prototype.hasOwnProperty.call(this.model.models[i],"W")){
                    // console.log("here")
                    let len_w = this.model.models[i].W.out.length
                    let len_b =  this.model.models[i].b.out.length

                    this.model.models[i].W.dout = utils.zeros(len_w);
                    this.model.models[i].b.dout = utils.zeros(len_b);
                }
            }

        }
    }

    function graph(model, id=null,draw=false){

        let data = Network(model);

        if(draw){
            let chart = anychart.graph(data)
            chart.container(id)
            chart.draw();
        }else{
            return data
        }

    }

    function Word2Vec(kwargs={}){

        let text = kwargs["text"]
        let stopwords = kwargs["stopwords"]
        this.embed_dim = kwargs["embed_dim"] || 50
        let window = kwargs["window"] || 5


        let text_lower = text.toLocaleLowerCase()
        
        let text_list = text_lower.split("\n")

        if(!stopwords){
            let stopwords = ["a","in","when","the","of","is","who"]
        }

        let [word_list, all_text] = word_utils.gen_word(window,text_list,stopwords);
        this.vocab = word_utils.unique_word(all_text)
        this.n_words = word_utils.obj_len(this.vocab);

        let [data, label] = word_utils.create_data(word_list, this.vocab,this.n_words)
        this.data = data
        this.label = label

    }

    Word2Vec.prototype = {

        train: function(epoch, lr=0.01){

            this.model = new Sequential([
                new Linear(this.n_words,this.embed_dim),
                new Linear(this.embed_dim,this.n_words),
                new Softmax()
            ]);

            let optim = new OptimSGD(this.model,lr=lr)

            for(let i=0; i< epoch; i++){

                let total_loss = 0;
                for(let j=0; j < this.data.length; j++){
            
                    let x_data = this.data[j]
                    let y_data = this.label[j]
            
                    let x = new Tensor(1,this.n_words, false);
                    x.setFrom(x_data)
                    x.name= "input"
            
                    this.model.forward(x)
            
                    // console.log(-Math.log(model.out.out[y_data-1]))
                    let loss = new Loss(y_data-1,this.model)
            
                    // console.log(loss.out);
                    total_loss += loss.out
            
                    loss.backward()
            
                    optim.step();
            
                    optim.grad_zero()
            
                }
            
                console.log(`for epoch ${i} Loss is ${total_loss/this.data.length}`)
            }
        },

        embed_weight: function(){

                let weight = this.model.models[0].get_weight()

                return weight;
        }
    }

    global.Scalar = Scalar;
    global.add = add;
    global.mult = mult;
    global.Mat = Mat;
    global.Tensor = Tensor;
    global.add = add;
    global.matmul = Matmul;
    global.Linear = Linear;
    global.ReLU = ReLU;
    global.Softmax = Softmax;
    global.Sequential = Sequential;
    global.Loss = Loss;
    global.OptimSGD = OptimSGD;
    global.graph = graph;
    global.Word2Vec = Word2Vec;

})(autograd)

module.exports = autograd
