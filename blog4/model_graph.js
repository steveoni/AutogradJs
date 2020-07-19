function Network(model){

    var data = {

        nodes: [],
        edges: []
    }

    let count = 0
    function recursive(m,name,root){

        // console.log("INPUT",name)
        if(m.hasOwnProperty("name")){
    
            if(m.name== "input"){
                data.nodes.push({"id": `${m.name}` })
                return ''
            }
            
        }
        
        
        // console.log(m.func_name);
        count +=1
        data.nodes.push({"id": `${m.func_name}`});

        if(m.hasOwnProperty("x") && m.hasOwnProperty("y")){
            root = m.func_name;
            data.edges.push({"from": `${root}`, "to": `${m.y.func_name || m.y.name}`})
            data.nodes.push({id:`${m.y.func_name || m.y.name}`})
            data.edges.push({"from": `${root}`, "to": `${m.x.func_name || m.x.name}`})
            return recursive(m.x, m.x.func_name,root)
        }

    
        if(m.hasOwnProperty("x")){
            root =  m.func_name;
            data.edges.push({"from": `${root}`, "to": `${m.x.func_name || m.x.name}`})
            return recursive(m.x, m.x.func_name,root)
        }
    

    
        if(m.hasOwnProperty("items")){
            root = m.func_name;
            data.edges.push({"from": `${root}`, "to": `${m.items.func_name || m.items.name}`})
            return recursive(m.items, m.items.func_name,root);
        }

    }

    recursive(model,null);

    return data

}
