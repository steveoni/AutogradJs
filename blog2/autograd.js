var utils = require("./utils.js");

var autograd = {};

(function(global){

    var Mat = function (n, d) {
        // n is number of rows d is number of columns
        this.n = n;
        this.d = d;
        this.out = utils.zeros(n * d);
        this.dout = utils.zeros(n * d);
    }

    var Tensor = function (n, d, require_grad) {

        this.n = n;
        this.d = d;
        this.out = utils.zeros(n * d);
        this.dout = utils.zeros(n * d);
        this.require_grad = require_grad;
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
            utils.assert(arr.length == this.n*this.d,"shape not compatible")
            for (var i = 0, n = arr.length; i < n; i++) {
                this.out[i] = arr[i];
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

    function add(x, y) {
        utils.assert(x.out.length === y.out.length);
    
    
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

    Matmul.prototype = {

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
    
            utils.assert(this.dout.length === g.length);
            this.dout = g;
    
        }
    }

    function Linear(in_dim,out_dim){

        this.W = new Tensor(in_dim,out_dim,true).randn(0,0.99);
        this.b = new Tensor(out_dim,1,true).randn(0,0.99);
    
        this.func_name = "<Linear>";
        this.require_grad = true;
    
    }

    Linear.prototype = {

        forward : function(x){
    
            this.mult = new Matmul(x,this.W)
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

    global.Mat = Mat;
    global.Tensor = Tensor;
    global.add = add;
    global.matmul = Matmul;
    global.Linear = Linear;
    global.ReLU = ReLU;
    global.Softmax = Softmax;
    
})(autograd)
