import { getConfig } from '../config/defaultConfig';
import useSWR from 'swr';
import { fetcher } from './user';

export const useDatasetDataDistribution = () => {
  const { data, mutate, error } = useSWR(
    getConfig()['environmentsDataDistributionUrl'],
    fetcher
  );
  const loadingDataDistribution = !data && !error;
  const environmentsDataDistribution = data && !error ? data : null;
  if (environmentsDataDistribution) {
    environmentsDataDistribution = environmentsDataDistribution.map(
      (environment) => {
        return { ...environment, id: environment._id };
      }
    );
  }
  return [
    environmentsDataDistribution,
    { mutateDataDistribution: mutate, loadingDataDistribution },
    error,
  ];
};

export const useDatasetTrainingDistribution = () => {
  const { data, mutate, error } = useSWR(
    getConfig()['environmentsTrainingDistributionUrl'],
    fetcher
  );
  const loadingTrainingDistribution = !data && !error;
  const environmentsTrainingDistribution = data && !error ? data : null;
  if (environmentsTrainingDistribution) {
    environmentsTrainingDistribution = environmentsTrainingDistribution.map(
      (environment) => {
        return { ...environment, id: environment._id };
      }
    );
  }
  return [
    environmentsTrainingDistribution,
    { mutateTrainigDistribution: mutate, loadingTrainingDistribution },
    error,
  ];
};
