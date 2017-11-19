// type argument like node ~.js <value>
var NUM_FILE = parseInt(process.argv[2]);
var chartWidth = 500, chartHeight = 500;


var fs = require('fs');
var d3 = require('d3');
var jsdom = require('jsdom');
var svg_to_png = require('svg-to-png');
var path = require('path')


var color = d3.scale.category10();

var dir = path.join(__dirname, 'svg');
if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
}
module.exports = function( pieData, outputLocation ){

      for (var i=(NUM_FILE * 100); i< (NUM_FILE+1) * 100; i++) {
	(function (i) {

	    jsdom.env({
		html:'',
		features:{ QuerySelector:true }, //you need query selector for D3 to work
		done:function(errors, window){
		    window.d3 = d3.select(window.document); //get d3 into the dom

        	var circle_object = {
         		 'class': 'circle',
         		 "r": function(d){ return d }
       		 };
		var rect_object = {
    	      		'class': 'rect',
      		    	"width": function(d){ return d },
       	  		 "height": function(d){ return d }
        	};
		sizeData = new Array();;
		for(var j=0;j<2;++j){
			sizeData.push((Math.random()+0.2)*100);
		}
        	//size1 = (Math.random()+0.2)*100; // radius : 20 ~ 120
        	//sizeData = [r];
        	outputLocation =  i + '.svg';

        
		    //do yr normal d3 stuff
		    var svg = window.d3.select('body')
			.append('div').attr('class','container') //make a container div to ease the saving process
			.append('svg')
			.attr({
			    xmlns:'http://www.w3.org/2000/svg',
			    width:chartWidth,
			    height:chartHeight
			})

		    for(var j=0;j<2;++j){
			// pull size data to fit # figures
			if(j==1)sizeData.pop();
 		    	var rand = Math.random();
			var scaler = Math.floor(rand * chartWidth);
        	    	rand = rand >= 0.5? 1 : 0;
			if(rand==0){
			    svg.selectAll('circle')
      		    	
				.data(sizeData)
      				.enter()
      		  	  	.append('circle')
				//.attr('transform','translate(' + Math.floor((Math.random()) * chartWidth) + ',' + Math.floor((Math.random()) * chartHeight) + ')') //set place where circle is.
				.attr("cx", function(d) {
  					return Math.floor((Math.random()) * chartWidth);
  				})
                                .attr("cy", function(d) {
                                        return Math.floor((Math.random()) * chartWidth);
                                })
				.attr(circle_object)
      			    	.style("fill", function() {
        				return "hsl(" + Math.random() * 360 + ",100%,50%)"; // set random color
      			    	});
		    	}else{
			    svg.selectAll('rect')

				.data(sizeData)
				.enter()
			    	.append('rect')
				//.attr('transform','translate(' + Math.floor((Math.random()) * chartWidth)  + ',' + Math.floor((Math.random()) * chartHeight)  + ')')
                    		.attr("x", function(d) {
                                        return Math.floor((Math.random()) * chartWidth);
                                })      
                                .attr("y", function(d) {
                                        return Math.floor((Math.random()) * chartWidth);
                                })
				.attr(rect_object)
                    		.style("fill", function() {
                        		return "hsl(" + Math.random() * 360 + ",100%,50%)"; // set random color
                    		});
			}
       		    }
	
		    //write out the children of the container div
		    fs.writeFileSync(path.join(__dirname, 'svg', outputLocation), window.d3.select('.container').html()) //using sync to keep the code simple
		    svg_to_png.convert(path.join(__dirname, 'svg', outputLocation), path.join(__dirname, "bitmap")).then(function() {

		    });
		    console.log(outputLocation)
		}
	    });


	})(i);

    }
}

if (require.main === module) {
    module.exports();
}

