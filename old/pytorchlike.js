


// Utility fun
function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}


var return_v = false;
var v_val = 0.0;
var gaussRandom = function () {
    if (return_v) {
        return_v = false;
        return v_val;
    }
    var u = 2 * Math.random() - 1;
    var v = 2 * Math.random() - 1;
    var r = u * u + v * v;
    if (r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2 * Math.log(r) / r);
    v_val = v * c; // cache this
    return_v = true;
    return u * c;
}
var randf = function (a, b) { return Math.random() * (b - a) + a; }
var randi = function (a, b) { return Math.floor(Math.random() * (b - a) + a); }
var randn = function (mu, std) { return mu + gaussRandom() * std; }

// helper function returns array of zeros of length n
// and uses typed arrays if available
var zeros = function (n) {
    if (typeof (n) === 'undefined' || isNaN(n)) { return []; }
    if (typeof ArrayBuffer === 'undefined') {
        // lacking browser support
        var arr = new Array(n);
        for (var i = 0; i < n; i++) { arr[i] = 0; }
        return arr;
    } else {
        return new Float64Array(n);
    }
}

var RandMat = function (n, d, mu, std) {
    var m = new Tensor(n, d);
    fillRandn(m, mu, std);
    //fillRand(m,-std,std); // kind of :P
    return m;
}

// Mat utils
// fill matrix with random gaussian numbers
var fillRandn = function (w, mu, std) { for (var i = 0, n = w.length; i < n; i++) { w[i] = randn(mu, std); } }

var Mat = function (n, d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.out = zeros(n * d);
    this.dout = zeros(n * d);
}
var Tensor = function (n, d, require_grad) {

    this.n = n;
    this.d = d;
    this.out = zeros(n * d);
    this.dout = zeros(n * d);
    this.require_grad = require_grad;
}

Tensor.prototype = {

    get: function (row, col) {

        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.out.length);
        return this.out[ix];

    },
    set: function (row, col, v) {
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.out.length);
        this.out[ix] = v;
    },
    setFrom: function (arr) {
        for (var i = 0, n = arr.length; i < n; i++) {
            this.out[i] = arr[i];
        }
    },
    randn: function (mu, std) {
        fillRandn(this.out, mu, std);
        return this;

    },
    grad: function (grad) {

        // for(var i=0,n=grad.length;i<n;i++){
        //     this.dout[i] = grad[i];
        // }
        this.dout = grad;
    }
}


function add(x, y) {
    assert(x.out.length === y.out.length);


    this.items = new Mat(1,x.out.length);
    for (var i = 0; i < x.out.length; i++) {

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

add.prototype = {

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

        assert(this.items.dout.length === g.length);
        this.dout = g;

    }
}

function Multid(x, y) {

    assert(x.d === y.n, "matmul dimension misaligned");

    this.n = x.n;
    this.d = y.d;
    this.x = x;
    this.y = y;
    this.require_grad = true;
    this.items = new Mat(this.n, this.d);
    this.out = this.items.out;
    this.dout = this.items.dout
    this.func_name = "<Multiply>";

    for (var i = 0; i < x.n; i++) {
        for (var j = 0; j < y.d; j++) {

            var dot = 0.0;
            for (var k = 0; k < x.d; k++) {

                dot += this.x.out[x.d * i + k] * this.y.out[y.d * k + j];
            }
            this.out[this.d * i + j] = dot;
        }
    }
}

Multid.prototype = {

    backward: function () {

        if (this.x.require_grad) {
            
            for(var i = 0;i< this.x.n;i++){
                for(var j=0;j<this.y.d;j++){
                    for(var k =0;k<this.x.d;k++){
                        var b = this.dout[this.y.d*i+j];
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

            for(var i = 0;i< this.x.n;i++){
                for(var j=0;j<this.y.d;j++){
                    for(var k =0;k<this.x.d;k++){
                        var b = this.dout[this.y.d*i+j];
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

        assert(this.dout.length === g.length);
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

        this.mult = new Multid(x,this.W)
        this.items = new add(this.mult,this.b);
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
        assert(this.dout.length === g.length);
        this.dout = g;
    },

    update: function(lr){

        for(var i=0;i< this.W.out.length;i++){

            this.W.out[i] -= (lr*this.W.dout[i]); 
        }
        for(var i=0;i< this.b.out.length;i++){

            this.b.out[i] -= (lr*this.b.dout[i]); 
        }



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
        assert(this.dout.length === g.length);

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

        var es = zeros(this.x.d);
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
}

Sequential.prototype = {

    forward : function(x){

        
        this.out = this.models[0].forward(x);

        for(var i=0;i<this.models.length;i++){

            if( i==0){
                continue;
            }
            this.out = this.models[i].forward(this.out);
        }
    },

}

function Loss(target, predict){

    this.model = predict;
    this.out =  - Math.log(predict.out[target]);
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

        for(var i in this.model.models){
            
            // console.log(model)
            if("update" in this.model.models[i]){
                // console.log("here")
                this.model.models[i].update(this.lr);
            }
        }
    }
}
var cv = new Tensor(1,2).randn(0, 0.008)


// var model = new Sequential([

//     new Linear(3,4),
//     new ReLU(),
//     new Linear(4,5),
//     new Softmax()
// ])

// var x = new Tensor(1,3).randn(0,0.001);
// model.forward(x)
// // model.models[model.models.length -1].backward(2)
// // model.out.backward(2)
// var optim = new OptimSGD(model,lr=0.001);
// var loss = new Loss(2,model.out);
// loss.backward()

// console.log("out",model.models[0].W.out,"dout",model.models[0].W.dout)
// // for(var i=0;i< model.models[0].W.out.length;i++){

// //     model.models[0].W.out[i] -= (0.001*model.models[0].W.dout[i]); 
// // }
// optim.step()
// console.log("after optim",model.models[0].W.out)

// var l1 = new Linear(3,4).forward(x);
// var r = new ReLU()

// r.forward(l1);

// var l = new Linear(4,5).forward(r);
// var s  = new Softmax().forward(l);
// s.backward(2);
// // var g = new Tensor(1,2).setFrom([2,6]);

// // l.grad(g.out);
// // l.backward()

// console.log(r.dout,l1.dout);




// var model = new Sequential([
//     new Linear(2,3),
//     new ReLU(),
//     new Linear(3,4),
//     new ReLU(),
//     new Linear(4,2),
//     new Linear(6,2),
//     new Softmax()
// ]);

// model.backward();
// //var optim = new SGD(model,Lr=0.01);
// var loss = new Loss(target,model);
// loss.backward();
// optim.step();


var data = [
    [0,0],
    [0,1],
    [1,0],
    [1,1],
    [0,2],
    [2,0],
    [2,2],
    [0,3],
    [3,0],
    [3,3],
    [0,4],
    [4,0],
    [4,4]
];

var y = [0,1,1,0,1,1,0,1,1,0,1,1,0];

var model = new Sequential([
    new Linear(2,3),
    new ReLU(),
    new Linear(3,2),
    new Softmax()
]);

var x = new Tensor(1,2,require_grad=true)
x.setFrom([2,3]);

var y = new Tensor(2,4,require_grad=false);
y.setFrom([2,3,4,5,6,7,8,1])

var m =model.forward(x)

var l = new Loss(1,model.out)
console.log(model.models[0].dout, l.out)
l.backward()
console.log(model.models[0].dout, l.out)
// model.forward(x)
// model.out.backward(1)
// console.log(x.dout);


// var optim = new OptimSGD(model,lr=0.001);
// var loss_sum = 0.0
// for(var i in data){

    
//     var x = new Tensor(1,2,require_grad=false)
//     x.setFrom(data[i]);
//     // console.log(x);

//     model.forward(x);

//     var loss = new Loss(y[i],model.out);

//     loss_sum +=loss.out;

//     console.log(loss.out);

//     loss.backward();

//     optim.step();
// }

// var x_t  = new Tensor(1,2,require_grad=false)
// x_t.setFrom([0,1]);
// model.forward(x_t)
// console.log(model.out);

