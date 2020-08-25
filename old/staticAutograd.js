const Network = require("./model_graph.js");
const utils = require("./utils.js")

function name(arg){

    if("name" in arg){
        arg.name += utils.randi(1,6)
    }else{
        arg.func_name += utils.randi(1,6)
    }
}

function Tensor(arr,require_grad){

    this.item = arr;
    this.require_grad = require_grad;
    this.gradv = 0;
    this.name = "<Tensor>";

}

Tensor.prototype = {

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
    this.func_name = "<Add>";

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



function multi(x, y) {

    this.item = x.item * y.item;
    this.x = x;
    this.y = y;

    name(x)
    name(y)
    this.gradv = 0;

    this.require_grad = true;
    this.func_name = "<Multi>"

}

multi.prototype = {

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


var x = new Tensor(-2,false);
var y = new Tensor(5,true);
var z = new Tensor(-4,true);

var q = new add(x,y);

var f = new multi(q,z);

f.grad(1)
f.backward()

console.log(new Network(f))