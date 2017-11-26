package neural;

import java.util.Arrays;
import java.util.Random;

public class Neural_lib {
	
	private double pattern [][];
	private double answer[][];
	private double eta = 0.2;
	private int hidden_number, output_number, learning_number;
	Random rand = new Random();
	
	private double w_hidden[][];
	private double bias_hidden[];
	private double w_output[][];
	private double bias_output[];
	/*
	private double w_hidden[][] = {
			{0.490,0.348,0.073,0.837,-0.071,-3.617,-0.536,-0.023,-1.717,-1.456,-0.556,0.852 },
			{0.442,-0.537,1.008,1.072,-0.733,0.823,-0.453,-0.014,-0.027,-0.427,1.876,-2.305 },
			{0.654,-1.389,1.246,0.057,-0.183,-0.743,-0.461,0.331,0.449,-1.296,1.569,-0.471}
	};
	private double bias_hidden[] = {-0.185, 0.526, -1.169};
	
	private double w_output[][] = {
			{0.388,0.803,0.029},
			{0.025,-0.790,1.553} 
	};
	private double bias_output[] = {-1.438,-1.379};
	*/
	
	private double z_hidden[];
	private double partial_hidden[][];
	private double bias_partial_hidden[];
	private double partial_output[][];
	private double bias_partial_output[];
	private double cost = 0; 
	
	public Neural_lib(double[][] pattern, double[][] answer, int hidden_number, int output_number, int learning_number) {
		this.pattern = pattern;
		this.answer = answer;
		this.hidden_number = hidden_number;
		this.output_number = output_number;
		this.learning_number = learning_number;
		w_hidden = new double[hidden_number][pattern[0].length];
		bias_hidden = new double[hidden_number];
		for(int i = 0; i < hidden_number; i++) {
			for(int j = 0; j < w_hidden[0].length; j++) {
				w_hidden[i][j] = rand.nextGaussian();
			}
			bias_hidden[i] = rand.nextGaussian();
		}
		w_output = new double[output_number][hidden_number];
		bias_output = new double[output_number];
		for(int i = 0; i < output_number; i++) {
			for(int j = 0; j < hidden_number; j++) {
				w_output[i][j] = rand.nextGaussian();
			}
			bias_output[i] = rand.nextGaussian();
		}
		z_hidden = new double[hidden_number];
		partial_hidden = new double[hidden_number][w_hidden[0].length];
		bias_partial_hidden = new double[hidden_number];
		partial_output = new double[output_number][w_output[0].length];
		bias_partial_output = new double[output_number];
	}
	
	public void getIteration() {
		for(int i = 0; i < learning_number; i++) {
			System.out.println("Learning_Number = "+i);
			this.getLearning();
		}
		System.out.println("Parameter");
		System.out.println("w_hidden = "+ Arrays.deepToString(w_hidden));
		System.out.println("bias_hidden = "+ Arrays.toString(bias_hidden));
		System.out.println("w_output = "+ Arrays.deepToString(w_output));
		System.out.println("bias_output = "+ Arrays.toString(bias_output));
	}

	public void getLearning() {
		double gradient_hidden[][] = new double[hidden_number][w_hidden[0].length];
		double gradient_bias_hidden[] = new double[hidden_number];
		double gradient_output[][] = new double[output_number][w_output[0].length];
		double gradient_bias_output[] = new double[output_number];
		cost = 0;
		for(int i = 0; i < pattern.length; i++) {
			//System.out.println("Learning = "+i);
			cost += this.getObject(i);
			//隠れ層勾配更新
			for(int j = 0; j < hidden_number; j++) {
				for(int k = 0; k < gradient_hidden[0].length; k++) {
					gradient_hidden[j][k] += partial_hidden[j][k]; 
				}
				gradient_bias_hidden[j] += bias_partial_hidden[j];
			}
			//出力層勾配更新
			for(int j = 0; j < output_number; j++) {
				for(int k = 0; k < gradient_output[0].length; k++) {
					gradient_output[j][k] += partial_output[j][k]; 
				}
				gradient_bias_output[j] += bias_partial_output[j];
			}
		}
		System.out.println("Cost = "+cost);
		//System.out.println("gradient_hidden = "+ Arrays.deepToString(gradient_hidden));
		//System.out.println("gradient_bias_hidden = "+ Arrays.toString(gradient_bias_hidden));
		//System.out.println("gradient_output = "+ Arrays.deepToString(gradient_output));
		//System.out.println("gradient_bias_output = "+ Arrays.toString(gradient_bias_output));
		
		//隠れ層:weight更新
		for(int i = 0; i < hidden_number; i++) {
			for(int j = 0; j < w_hidden[0].length; j++) {
				w_hidden[i][j] -= eta * gradient_hidden[i][j];
			}
			bias_hidden[i] -= eta * gradient_bias_hidden[i];
		}
		//出力層:weight更新
		for(int i = 0; i < output_number; i++) {
			for(int j = 0; j < w_output[0].length; j++) {
				w_output[i][j] -= eta * gradient_output[i][j];
			}
			bias_output[i] -= eta * gradient_bias_output[i];
		}
		//System.out.println("Parameter");
		//System.out.println("w_hidden = "+ Arrays.deepToString(w_hidden));
		//System.out.println("bias_hidden = "+ Arrays.toString(bias_hidden));
		//System.out.println("w_output = "+ Arrays.deepToString(w_output));
		//System.out.println("bias_output = "+ Arrays.toString(bias_output));
	}
	
	public double getObject(int pattern_id) {
		double z_hidden[] = new double[this.hidden_number];
		double a_hidden[] = new double[this.hidden_number];
		double da_hidden[] = new double[this.hidden_number];
		double z_output[] = new double[this.output_number];
		double a_output[] = new double[this.output_number];
		double da_output[] = new double[this.output_number];
		double delta_output[] = new double[this.output_number];
		double delta_hidden[] = new double[this.hidden_number];
		double c = 0;
		
		for(int i = 0; i < this.hidden_number; i++) {
			z_hidden[i] = this.getInnerProduct(w_hidden[i], pattern[pattern_id]) + bias_hidden[i];
			a_hidden[i] = 1 / (1 + Math.exp( - z_hidden[i]));
			da_hidden[i] = a_hidden[i] * (1 - a_hidden[i]);
		}
		for(int i = 0; i < this.output_number; i++) {
			z_output[i] = this.getInnerProduct(w_output[i], a_hidden) + bias_output[i];
			a_output[i] = 1 / (1 + Math.exp( - z_output[i]));
			da_output[i] = a_output[i] * (1 - a_output[i]);
		}
		for(int i = 0; i < output_number; i++) {
			c += Math.pow(a_output[i] - answer[pattern_id][i], 2);
		}
		c /= 2;
		
		//誤差算出:出力層
		double wt_output[][] = this.getTranspose(w_output);
		for(int i = 0; i < output_number; i++) {
			delta_output[i] = (a_output[i] - answer[pattern_id][i]) * da_output[i];
		}
		//誤差算出:隠れ層
		for(int i = 0; i < hidden_number; i++) {
			delta_hidden[i] = this.getInnerProduct(delta_output,wt_output[i]) * da_hidden[i];
		}
		
		//二乗誤差の偏微分:隠れ層
		for(int i = 0; i < hidden_number; i++) {
			for(int j = 0; j < partial_hidden[0].length; j++) {
				partial_hidden[i][j] = delta_hidden[i] * pattern[pattern_id][j];
			}
			bias_partial_hidden[i] = delta_hidden[i];
		}
		
		//二乗誤差の偏微分:出力層
		for(int i = 0; i < output_number; i++) {
			for(int j = 0; j < partial_output[0].length; j++) {
				partial_output[i][j] = delta_output[i] * a_hidden[j];
			}
			bias_partial_output[i] = delta_output[i];
		}
		
		//System.out.println("z_hidden = "+ Arrays.toString(z_hidden));
		//System.out.println("a_hidden = "+ Arrays.toString(a_hidden));
		//System.out.println("da_hidden = "+ Arrays.toString(da_hidden));
		//System.out.println("z_output = "+ Arrays.toString(z_output));
		//System.out.println("a_output = "+ Arrays.toString(a_output));
		//System.out.println("da_output = "+ Arrays.toString(da_output));
		//System.out.println("c = "+ c);
		//System.out.println("delta_output = "+ Arrays.toString(delta_output));
		//System.out.println("delta_hidden = "+ Arrays.toString(delta_hidden));
		//System.out.println("partial_hidden = "+ Arrays.deepToString(partial_hidden));
		//System.out.println("bias_partial_hidden = "+ Arrays.toString(bias_partial_hidden));
		//System.out.println("partial_output = "+ Arrays.deepToString(partial_output));
		//System.out.println("bias_partial_output = "+ Arrays.toString(bias_partial_output));
		return c;
	}
	
	//テスト
	public double[] getTest(double testpattern[]) {
		double z_hidden[] = new double[this.hidden_number];
		double a_hidden[] = new double[this.hidden_number];
		double z_output[] = new double[this.output_number];
		double a_output[] = new double[this.output_number];
		
		for(int i = 0; i < this.hidden_number; i++) {
			z_hidden[i] = this.getInnerProduct(w_hidden[i], testpattern) + bias_hidden[i];
			a_hidden[i] = 1 / (1 + Math.exp( - z_hidden[i]));
		}
		for(int i = 0; i < this.output_number; i++) {
			z_output[i] = this.getInnerProduct(w_output[i], a_hidden) + bias_output[i];
			a_output[i] = 1 / (1 + Math.exp( - z_output[i]));
		}
		System.out.println("Result");
		System.out.println("Test_a_output = "+ Arrays.toString(a_output));
		return a_output;
	}
	
	//内積
	public double getInnerProduct(double a[], double b[]) {
		double answer = 0;
		for(int i = 0; i < a.length; i++) {
			answer += a[i] * b[i];
		}
		return answer;
	}
	//転置行列
	public double[][] getTranspose(double [][]a){
		double t[][] = new double[a[0].length][a.length];
		for(int i = 0; i < a[0].length; i++) {
			for(int j = 0; j < a.length; j++) {
				t[i][j] = a[j][i];
			}
		}
		return t;
	}

}
