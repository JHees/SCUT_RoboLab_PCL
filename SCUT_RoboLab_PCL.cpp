#include<iostream>
#include<boost/progress.hpp>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/filters/passthrough.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/radius_outlier_removal.h>
#include<pcl/filters/statistical_outlier_removal.h>
//#include<pcl/filters/conditional_removal.h>

#include<pcl/sample_consensus/sac_model_plane.h>
#include<pcl/sample_consensus/ransac.h>

#include<pcl/kdtree/kdtree_flann.h>
#include<pcl/segmentation/extract_clusters.h>

#include<pcl/segmentation/sac_segmentation.h>

#include<pcl/common/centroid.h>

#include<pcl/visualization/cloud_viewer.h>
#include<stdlib.h>

using PointT = pcl::PointXYZRGB;
int main()
{
  boost::progress_timer t;
  boost::shared_ptr<pcl::PointCloud<PointT> > cloud(new pcl::PointCloud<PointT>);
  if (pcl::io::loadPCDFile<PointT> ("../resource/2.pcd", *cloud) == -1) 
  {
    PCL_ERROR ("Could not read file test.pcd \n");
    return (-1);
  }

pcl::PassThrough<PointT> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.5,1);
  pass.filter(*cloud);
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(-1,0);
  pass.filter(*cloud);

// pcl::ConditionAnd<PointT>::Ptr cond(new pcl::ConditionAnd<PointT>);
//   cond->addComparison(pcl::PackedRGBComparison<PointT>::ConstPtr (new pcl::PackedRGBComparison<PointT>("b",pcl::ComparisonOps::GT,128)));
//   cond->addComparison(pcl::PackedRGBComparison<PointT>::ConstPtr (new pcl::PackedRGBComparison<PointT>("r",pcl::ComparisonOps::GE,128)));
//   cond->addComparison(pcl::PackedRGBComparison<PointT>::ConstPtr (new pcl::PackedRGBComparison<PointT>("g",pcl::ComparisonOps::GE,128)));
//   pcl::ConditionalRemoval<PointT> cr;
//   cr.setInputCloud(cloud);
//   cr.setCondition(cond);
//   cr.setKeepOrganized(true);
//   cr.filter(*cloud);

pcl::VoxelGrid<PointT> grid;
  grid.setInputCloud(cloud);
  grid.setLeafSize(0.01,0.01,0.05);
  grid.filter(*cloud);

pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(0.01);
  sor.filter(*cloud);

pcl::RadiusOutlierRemoval<PointT> outlier;
  outlier.setInputCloud(cloud);
  outlier.setRadiusSearch(0.1);
  outlier.setMinNeighborsInRadius(100);
  outlier.filter(*cloud);


pcl::SampleConsensusModelPlane<PointT>::Ptr plane(new pcl::SampleConsensusModelPlane<PointT>(cloud));
  std::vector<int> inliers;
  pcl::RandomSampleConsensus<PointT> ransac(plane);
  ransac.setDistanceThreshold(0.01);
  ransac.computeModel();
  ransac.getInliers(inliers);
  pcl::copyPointCloud(*cloud,inliers,*cloud);


pcl::EuclideanClusterExtraction<PointT> ec;
  pcl::search::KdTree<PointT> ::Ptr kdtree(new pcl::search::KdTree<PointT>);
  std::vector<pcl::PointIndices> indics;
  kdtree->setInputCloud(cloud);
  ec.setClusterTolerance(0.1);
  ec.setMinClusterSize(200);
  ec.setInputCloud(cloud);
  ec.extract(indics);
  // std::vector<pcl::PointCloud<PointT>::Ptr> cloud_Clu;
  // for(auto i:indics)
  // {
  //   pcl::PointCloud<PointT> ::Ptr buf(new pcl::PointCloud<PointT>);
  //   pcl::copyPointCloud(*cloud,i.indices,*buf);
  //   cloud_Clu.push_back(buf);
  // }


pcl::SACSegmentation<PointT> seg;
  std::vector<pcl::ModelCoefficients::Ptr> coeff;
  std::vector<pcl::PointIndices::Ptr> inl;
  std::vector<Eigen::Vector4f> centr;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);
  seg.setInputCloud(cloud);
  pcl::PointIndices::Ptr buf(new pcl::PointIndices);
  for(auto i:indics)
  {
      pcl::ModelCoefficients::Ptr buf_c(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr buf_i(new pcl::PointIndices);
      Eigen::Vector4f buf_centr;
      *buf=i;
      seg.setIndices(buf);
      seg.segment(*buf_i,*buf_c);
      pcl::compute3DCentroid(*cloud,i,buf_centr);
      inl.push_back(buf_i);
      coeff.push_back(buf_c);
      centr.push_back(buf_centr);
  }
  for(size_t i=0;i<inl.size();++i)
  {
      std::cout<<"Matched surface: "<<i<<"  -  "<<coeff[i]->values[0]<<" "<<coeff[i]->values[1]<<" "<<coeff[i]->values[2]<<" "<<coeff[i]->values[3]
                        <<" - ("<<centr[i][0]<<  ","<<centr[i][1]<<  ","<<centr[i][2]<<  ")"
                        <<" - "<<inl[i]->indices.size()<<std::endl;
  }


std::cout<<t.elapsed()<<std::endl;
pcl::PointIndices allPoints;
  for(auto i:indics)
  {
    allPoints.indices.insert(allPoints.indices.end(),i.indices.begin(),i.indices.end());
  }
  pcl::copyPointCloud(*cloud,allPoints,*cloud);
pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.initCameraParameters();
  viewer.setCameraPosition(0,-0.05,0.01,0,-1,0);
  viewer.addCoordinateSystem(1);
  viewer.addPointCloud(cloud);

  while (!viewer.wasStopped())
    viewer.spinOnce(100);

}