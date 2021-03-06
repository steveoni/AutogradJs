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
            this.x.grad(this.y.item * this.gradv);
            if ("backward" in this.x) {
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
