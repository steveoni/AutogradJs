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
    this.require_grad=true;

    this.item = x.item + y.item
    this.gradv = 0;
    this.name = "<Add>";

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

    this.gradv = 0;

    this.require_grad = true;
    this.name = "<Multi>"

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

// var arr = new Tensor(9,true);

// var arr2 = new Tensor(12,require_grad=true);
// var ad = new multi(arr,arr2);

// // var cont = new Tensor(10,false);

// // var mul = new multi(cont,ad)
// // console.log("arr2 grad b4 baclprop",ad.gradv)
// // console.log("ad", mul.item)
// ad.backward()

var x = new Tensor(-2,false);
var y = new Tensor(5,true);
var z = new Tensor(-4,true);

var q = new add(x,y);

var f = new multi(q,z);

f.grad(1)
f.backward()
console.log(f);