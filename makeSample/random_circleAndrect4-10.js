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

        	  outputLocation =  i + '.svg';

    		    //do yr normal d3 stuff
    		    var svg = window.d3.select('body')
      			.append('div').attr('class','container') //make a container div to ease the saving process
      			.append('svg')
      			.attr({
      			    xmlns:'http://www.w3.org/2000/svg',
      			    width:chartWidth,
      			    height:chartHeight
      			});

            var temp_arr = [4,7,10];
            //rand is 4 or 7 or 10
            var rand = temp_arr[Math.floor(Math.random() * temp_arr.length)];
            //temp_rand is in the range of [0,rand]
            var temp_rand = Math.floor(Math.random() * (rand+1))

            var circleData = new Array();
            for(var j=0;j<temp_rand;++j){
              var temp = {};
              temp['radius'] = (Math.random()+0.2)*100;
              temp['color'] = Math.floor(Math.random()*13);
              circleData.push(temp);
            }
            var rectData = new Array();
            for(var j=0;j<rand - temp_rand;++j){
              var temp = {};
              temp['width'] = (Math.random()+0.2)*100;
              temp['height'] = (Math.random()+0.2)*100;
              temp['color'] = Math.floor(Math.random()*13);
              rectData.push(temp);
            }

  			    svg.selectAll('circle')
                 .data(circleData)
            		 .enter()
            		 .append('circle')
      				   .attr("cx", function(d) {
        				       return Math.floor((Math.random()) * chartWidth);
        				 })
                 .attr("cy", function(d) {
                    return Math.floor((Math.random()) * chartWidth);
                 })
      				   .attr({
                		'class': 'circle',
                		"r": function(d){ return d.radius }
            		  })
      			     .style("fill", function(d) {
                    return "hsl(" + 30 * d.color + ",100%,50%)"; // set random color
                 });

    			  svg.selectAll('rect')
    				     .data(rectData)
    				     .enter()
    			    	 .append('rect')
              	 .attr("x", function(d) {
                    return Math.floor((Math.random()) * chartWidth);
                 })
                 .attr("y", function(d) {
                    return Math.floor((Math.random()) * chartWidth);
                 })
    				     .attr({
           	        'class': 'rect',
             		    "width": function(d){ return d.width },
              	  	"height": function(d){ return d.height }
                 })
                 .style("fill", function(d) {
             		    return "hsl(" + 30 * d.color + ",100%,50%)"; // set random color
            		 });

    		    //write out the children of the container div
    		    fs.writeFileSync(path.join(__dirname, 'svg', outputLocation), window.d3.select('.container').html()) //using sync to keep the code simple
    		    svg_to_png.convert(path.join(__dirname, 'svg', outputLocation), path.join(__dirname, "bitmap")).then(function() {});
    		    console.log(outputLocation)
    		}
	    })
  	})(i)
  }
}

if (require.main === module) {
    module.exports();
}
