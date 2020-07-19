


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
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
}
var Tensor = function (n, d, require_grad) {

    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
    this.require_grad = true;
}

Tensor.prototype = {

    get: function (row, col) {

        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        return this.w[ix];

    },
    set: function (row, col, v) {
        var ix = (this.d * row) + col;
        assert(ix >= 0 && ix < this.w.length);
        this.w[ix] = v;
    },
    setFrom: function (arr) {
        for (var i = 0, n = arr.length; i < n; i++) {
            this.w[i] = arr[i];
        }
    },
    randn: function (mu, std) {
        fillRandn(this.w, mu, std);
        return this;

    },
    grad: function (grad) {

        // for(var i=0,n=grad.length;i<n;i++){
        //     this.dw[i] = grad[i];
        // }
        this.dw = grad;
    }
}


function add(x, y) {
    assert(x.w.length === y.w.length);


    this.items = new Mat(1,x.w.length);
    for (var i = 0; i < x.w.length; i++) {

        this.items.w[i] = x.w[i] + y.w[i];
    }
    this.x = x;
    this.y = y;
    this.require_grad = true;
    this.w = this.items.w;
    this.dw = this.items.dw;
    this.n = this.items.n;
    this.d = this.items.d;

    // this.gradv = 1;

}

add.prototype = {

    backward: function () {

        if (this.x.require_grad) {
            this.x.grad(this.dw);
            if ("backward" in this.x) {
                this.x.backward()
            }


        }

        if (this.y.require_grad) {
            this.y.grad(this.dw);
            if ("backward" in this.y) {
                this.y.backward()
            }
        }
    },

    grad: function (g) {

        assert(this.items.dw.length === g.length);
        this.dw = g;

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
    this.w = this.items.w;
    this.dw = this.items.dw

    for (var i = 0; i < x.n; i++) {
        for (var j = 0; j < y.d; j++) {

            var dot = 0.0;
            for (var k = 0; k < x.d; k++) {

                dot += this.x.w[x.d * i + k] * this.y.w[y.d * k + j];
            }
            this.w[this.d * i + j] = dot;
        }
    }
}

Multid.prototype = {

    backward: function () {

        if (this.x.require_grad) {
            
            for(var i = 0;i< this.x.n;i++){
                for(var j=0;j<this.y.d;j++){
                    for(var k =0;k<this.x.d;k++){
                        var b = this.dw[this.y.d*i+j];
                        // console.log(b);
                        this.x.dw[this.x.d*i+k] += this.y.w[this.y.d*k+j] * b;

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
                        var b = this.dw[this.y.d*i+j];
                        this.y.dw[this.y.d*k+j] += this.x.w[this.x.d*i+k] * b;

                    }
                }
            }      

            if ("backward" in this.y) {
                this.y.backward()
            }
        }
    },

    grad: function (g) {

        assert(this.dw.length === g.length);
        this.dw = g;

    }
}


function Linear(x,in_dim,out_dim){

    this.W = new Tensor(in_dim,out_dim).randn(0,0.008);
    this.b = new Tensor(out_dim,1).randn(0,0.008);

    this.mult = new Multid(x,this.W)
    this.items = new add(this.mult,this.b);
    this.n = this.mult.n;
    this.d = this.mult.d;
    this.w = this.items.w;
    this.dw = this.items.dw;

    this.require_grad = true;

}

Linear.prototype = {

    backward: function(){

            this.items.grad(this.dw);
            this.items.backward();
    },
    grad: function(g){
        assert(this.dw.length === g.length);
        this.dw = g;
    }
}

function ReLU(x){

    this.items = new Mat(x.n,x.d);
    this.x = x;
    this.n = x.n;
    this.d = x.d;
    this.require_grad = true;
    for(var i=0;i<x.w.length;i++){

        this.items.w[i] = Math.max(0,x.w[i]);
    }
    this.w = this.items.w;
    this.dw = this.items.dw;

}
ReLU.prototype = {

    backward: function(){
        
        
        for(var i=0;i<this.x.w.length;i++){

            this.x.dw[i] = this.x.w[i] > 0 ? this.dw[i] : 0.0;
        }
        this.x.backward();


    },
    grad: function(g){
        assert(this.dw.length === g.length);

        this.dw = g;

    }
}


function Softmax(x){

        this.items = new Mat(1,x.d)

        this.x = x;
        //compute max activation
        var as = x.w;
        var amax = x.w[0];
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
            this.items.w[i] = es[i];
        }
        this.w = es; // saved for backprop

}
Softmax.prototype = {

    backward: function(y){

        var x = this.x;
        for(var i=0;i<this.items.d;i++){

            var indicator = i === y ? 1.0 : 0.0;
            var mul = -(indicator - this.w[i])
            x.dw[i] = mul;
        }

        this.x.grad(x.dw)//there is no need for this. we initiallizing dw twice
        // this.x.backward();
    }
}


function Loss(target, predict){

    this.w =  - Math.log(predicted[target]);
    this.y = target;
}

Loss.prototype = {

    backward : function(){

        
    }
}
var cv = new Tensor(1,2).randn(0, 0.008)

var soft = new Softmax(cv);

soft.backward(1)


var x = new Tensor(3,3,require_grad=true)
x.setFrom([2,3,4,5,6,7,8,9,10]);
console.log(x.w[3*0+0]);
